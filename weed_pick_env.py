"""Gymnasium MuJoCo environment for a Franka weed pick task.

Scope of this version:
- proprioception + relative weed position
- 6D delta end-effector action + continuous gripper-width action
- correct qpos vs DoF indexing
- damped least-squares differential IK
- bilateral grasp detection
- pick-only success (grasp and lift)
- small, reset-time domain randomization


"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dm_control import mjcf

from new2 import build_scene


@dataclass(frozen=True)
class SceneNames:
    ee_site: str = "panda/umi_gripper/ee_site"
    pot_joint: str = "pot_freejoint"
    weed_body: str = "plant"
    weed_joint: str = "weed_freejoint"
    arm_joints: tuple[str, ...] = tuple(f"panda/joint{i}" for i in range(1, 8))
    arm_actuators: tuple[str, ...] = tuple(f"panda/actuator{i}" for i in range(1, 8))
    finger_joints: tuple[str, str] = (
        "panda/umi_gripper/left_finger_joint",
        "panda/umi_gripper/right_finger_joint",
    )
    gripper_actuator: str = "panda/umi_gripper/fingers_actuator"
    left_pad_geoms: tuple[str, ...] = ("panda/umi_gripper/left_finger_pad",)
    right_pad_geoms: tuple[str, ...] = ("panda/umi_gripper/right_finger_pad",)
    weed_geoms: tuple[str, ...] = ("weed_geom",)


DEFAULT_DR_CONFIG: dict[str, Any] = {
    "pot_xy_range": 0.03,
    "weed_yaw_range": np.pi,
    "robot_qpos_noise": 0.01,
    "action_noise_std": 0.0,
    # Kept deliberately narrow for the first working version.
    "weed_mass_scale_range": (0.95, 1.05),
    "weed_friction_scale_range": (0.9, 1.1),
}


HOME_ARM = np.array([0.0, -0.405, 0.0, -2.25, 0.0, 1.28, 1.0], dtype=np.float64)
DEFAULT_GRIPPER_OPENING = 0.03


class WeedPickEnv(gym.Env[np.ndarray, np.ndarray]):
    """Pick-only Franka environment.

    Action layout, all normalized to [-1, 1]:
        [dx, dy, dz, dRx, dRy, dRz, gripper_width]

    The last component is continuous. -1 maps to minimum physical gripper
    width and +1 maps to maximum physical gripper width.
    """

    metadata = {"render_modes": []}

    OBS_COMPONENTS = {
        "arm_qpos": 7,
        "arm_qvel": 7,
        "ee_pos": 3,
        "ee_quat": 4,
        "gripper_width": 1,
        "weed_rel_ee": 3,
        "last_action": 7,
    }
    OBS_DIM = sum(OBS_COMPONENTS.values())

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        names: SceneNames | None = None,
        dr_config: dict[str, Any] | None = None,
        home_arm: np.ndarray = HOME_ARM,
        max_episode_steps: int = 150,
        frame_skip: int = 10,
        pos_scale: float = 0.015,
        rot_scale: float = np.deg2rad(4.0),
        ik_damping: float = 0.05,
        lift_height: float = 0.05,
    ) -> None:
        super().__init__()

        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        self.names = names or SceneNames()
        self._physics: mjcf.Physics | None = None
        if xml_path is None:
            scene = build_scene(export_xml=False)
            self._physics = mjcf.Physics.from_mjcf_model(scene)
            self.model = self._physics.model.ptr
            self.data = self._physics.data.ptr
        else:
            scene_path = Path(xml_path).expanduser().resolve()
            if not scene_path.is_file():
                raise FileNotFoundError(f"MJCF file not found: {scene_path}")
            self.model = mujoco.MjModel.from_xml_path(str(scene_path))
            self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.ik_damping = float(ik_damping)
        self.lift_height = float(lift_height)

        self.dr_config = deepcopy(DEFAULT_DR_CONFIG)
        if dr_config is not None:
            self.dr_config.update(deepcopy(dr_config))

        self._cache_model_ids()
        self._validate_actuator_layout()

        self._nominal_arm_qpos = np.asarray(home_arm, dtype=np.float64).copy()
        if self._nominal_arm_qpos.shape != (7,):
            raise ValueError("home_arm must contain exactly seven joint positions.")
        self._validate_home_pose()
        self._nominal_weed_qpos = self.model.qpos0[
            self.weed_qpos_adr : self.weed_qpos_adr + 7
        ].copy()
        self._nominal_pot_qpos = self.model.qpos0[
            self.pot_qpos_adr : self.pot_qpos_adr + 7
        ].copy()
        self._weed_rest_height = float(self._weed_position()[2])
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()

        # Gripper command is action[6].
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._prev_action = np.zeros(7, dtype=np.float32)
        self._last_action_delta = np.zeros(7, dtype=np.float32)

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._prev_action.fill(0.0)
        self._last_action_delta.fill(0.0)

        self._restore_nominal_model_parameters()
        self._apply_domain_randomization()
        self._randomize_pot_and_weed_pose()
        self._randomize_robot_qpos()
        self.data.qpos[self.finger_qpos_adr] = DEFAULT_GRIPPER_OPENING  ##IN here the diplsvment is put on both fingers
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_adr]
        self.data.ctrl[self.gripper_actuator_id] = DEFAULT_GRIPPER_OPENING
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        executed_action = action.copy()
        noise_std = float(self.dr_config["action_noise_std"])
        if noise_std > 0.0:
            executed_action += self.np_random.normal(0.0, noise_std, size=7)
            executed_action = np.clip(executed_action, -1.0, 1.0)

        arm_target = self._ee_delta_to_qpos_target(executed_action[:6])
        gripper_target = self._normalized_to_gripper_target(float(executed_action[6]))

        self.data.ctrl[self.arm_actuator_ids] = arm_target
        self.data.ctrl[self.gripper_actuator_id] = gripper_target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._last_action_delta = executed_action - self._prev_action
        self._prev_action = executed_action.copy()
        self._step_count += 1

        info = self._get_info()
        reward = self._compute_reward(info)
        terminated = bool(info["success"] or info["failure"])
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        return self._get_obs(), reward, terminated, truncated, info

    def _cache_model_ids(self) -> None:
        self.ee_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, self.names.ee_site
        )
        self.pot_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT, self.names.pot_joint
        )
        self.weed_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, self.names.weed_body
        )
        self.weed_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT, self.names.weed_joint
        )

        if self.model.jnt_type[self.pot_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The pot joint must be a MuJoCo free joint.")
        self.pot_qpos_adr = int(self.model.jnt_qposadr[self.pot_joint_id])

        if self.model.jnt_type[self.weed_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The weed joint must be a MuJoCo free joint.")
        self.weed_qpos_adr = int(self.model.jnt_qposadr[self.weed_joint_id])

        self.arm_joint_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.names.arm_joints
            ],
            dtype=np.int32,
        )
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids].astype(np.int32)
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_ids].astype(np.int32)

        self.finger_joint_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.names.finger_joints
            ],
            dtype=np.int32,
        )
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_ids].astype(
            np.int32
        )

        self.arm_actuator_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.names.arm_actuators
            ],
            dtype=np.int32,
        )
        self.gripper_actuator_id = self._require_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.names.gripper_actuator
        )

        self.left_pad_geom_ids = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.names.left_pad_geoms
        }
        self.right_pad_geom_ids = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.names.right_pad_geoms
        }
        self.weed_geom_ids = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.names.weed_geoms
        }

    def _validate_actuator_layout(self) -> None:
        if len(self.arm_actuator_ids) != 7:
            raise ValueError("Expected exactly seven Franka arm actuators.")

        # This environment sends joint-position targets to these actuators.
        for actuator_id in self.arm_actuator_ids:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id not in self.arm_joint_ids:
                raise ValueError(
                    f"Arm actuator {actuator_id} is not attached to a configured arm joint."
                )

    def _validate_home_pose(self) -> None:
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        if np.any(self._nominal_arm_qpos < lower) or np.any(self._nominal_arm_qpos > upper):
            raise ValueError("The configured Panda home pose violates a joint limit.")

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"Missing MuJoCo object {name!r} of type {obj_type.name}")
        return obj_id

    def _ee_delta_to_qpos_target(self, ee_delta: np.ndarray) -> np.ndarray:
        dpos = np.asarray(ee_delta[:3], dtype=np.float64) * self.pos_scale
        drot = np.asarray(ee_delta[3:], dtype=np.float64) * self.rot_scale
        twist = np.concatenate((dpos, drot))

        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jac_pos,
            jac_rot,
            self.ee_site_id,
        )
        jac_arm = np.vstack((jac_pos, jac_rot))[:, self.arm_dof_adr]

        damping_sq = self.ik_damping**2
        jac_pinv = jac_arm.T @ np.linalg.solve(
            jac_arm @ jac_arm.T + damping_sq * np.eye(6),
            np.eye(6),
        )
        dq_task = jac_pinv @ twist

        # Keep the redundant seventh DoF near a comfortable reset posture.
        q = self.data.qpos[self.arm_qpos_adr].copy()
        dq_posture = 0.05 * (self._nominal_arm_qpos - q)
        null_projector = np.eye(7) - jac_pinv @ jac_arm
        q_target = q + dq_task + null_projector @ dq_posture

        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        return np.clip(q_target, lower + 0.01, upper - 0.01)

    def _normalized_to_gripper_target(self, command: float) -> float:
        """Map normalized action [-1, 1] to actuator control range."""
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        alpha = 0.5 * (float(np.clip(command, -1.0, 1.0)) + 1.0)
        return float(ctrl_min + alpha * (ctrl_max - ctrl_min))

    def _current_gripper_width(self) -> float:
        # Assumes symmetric finger qpos values whose sum is physical opening.
        return float(np.sum(self.data.qpos[self.finger_qpos_adr]))

    def _get_obs(self) -> np.ndarray:
        arm_qpos = self.data.qpos[self.arm_qpos_adr].copy()
        arm_qvel = self.data.qvel[self.arm_dof_adr].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_quat = self._site_xquat(self.ee_site_id)
        if ee_quat[0] < 0.0:
            ee_quat = -ee_quat

        weed_pos = self._weed_position()
        obs = np.concatenate(
            (
                arm_qpos,
                arm_qvel,
                ee_pos,
                ee_quat,
                np.array([self._current_gripper_width()]),
                weed_pos - ee_pos,
                self._prev_action,
            )
        ).astype(np.float32)

        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation has shape {obs.shape}; expected {self.observation_space.shape}"
            )
        return obs

    def _weed_position(self) -> np.ndarray:
        geom_ids = np.fromiter(self.weed_geom_ids, dtype=np.int32)
        return np.mean(self.data.geom_xpos[geom_ids], axis=0)  ## NOt nessarily weed body origin, as each geometry is weighted equally, but should be close enough

    def _site_xquat(self, site_id: int) -> np.ndarray:
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return quat

    def _is_grasping(self) -> bool:
        left_contact = False
        right_contact = False

        for contact in self.data.contact[: self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.weed_geom_ids:
                left_contact |= bool(pair & self.left_pad_geom_ids)
                right_contact |= bool(pair & self.right_pad_geom_ids)
            if left_contact and right_contact:
                return True
        return False

    def _get_info(self) -> dict[str, Any]:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        weed_pos = self._weed_position()
        reach_distance = float(np.linalg.norm(weed_pos - ee_pos))
        is_grasped = self._is_grasping()
        lift_amount = float(weed_pos[2] - self._weed_rest_height)
        success = bool(is_grasped and lift_amount >= self.lift_height)

        # Simple failure guard for obviously invalid episodes.
        failure = bool(not np.all(np.isfinite(self.data.qpos)) or weed_pos[2] < -0.1)
        return {
            "success": success,
            "is_success": success,
            "failure": failure,
            "is_grasped": is_grasped,
            "reach_distance": reach_distance,
            "lift_amount": lift_amount,
            "control_dt": self.control_dt,
        }

    def _compute_reward(self, info: dict[str, Any]) -> float:
        reach_reward = 1.0 - np.tanh(8.0 * info["reach_distance"])
        reward = float(reach_reward)

        if info["is_grasped"]:
            lift_progress = np.clip(info["lift_amount"] / self.lift_height, 0.0, 1.0)
            reward = 2.0 + float(lift_progress)

        reward -= 0.002 * float(np.sum(np.square(self._last_action_delta)))
        if info["success"]:
            reward = 4.0
        if info["failure"]:
            reward = -1.0
        return reward

    def _restore_nominal_model_parameters(self) -> None:
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.geom_friction[:] = self._nominal_geom_friction

    def _apply_domain_randomization(self) -> None:
        mass_scale = self.np_random.uniform(
            *self.dr_config["weed_mass_scale_range"]
        )
        self.model.body_mass[self.weed_body_id] = (
            self._nominal_body_mass[self.weed_body_id] * mass_scale
        )

        friction_scale = self.np_random.uniform(
            *self.dr_config["weed_friction_scale_range"]
        )
        for geom_id in self.weed_geom_ids:
            self.model.geom_friction[geom_id] = (
                self._nominal_geom_friction[geom_id] * friction_scale
            )

    def _randomize_pot_and_weed_pose(self) -> None:
        xy_range = float(self.dr_config["pot_xy_range"])
        yaw_range = float(self.dr_config["weed_yaw_range"])

        xy_offset = self.np_random.uniform(-xy_range, xy_range, size=2)
        pot_qpos = self._nominal_pot_qpos.copy()
        weed_qpos = self._nominal_weed_qpos.copy()
        pot_qpos[:2] += xy_offset
        weed_qpos[:2] += xy_offset

        yaw = self.np_random.uniform(-yaw_range, yaw_range)
        weed_qpos[3:7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        )

        self.data.qpos[self.pot_qpos_adr : self.pot_qpos_adr + 7] = pot_qpos
        self.data.qpos[self.weed_qpos_adr : self.weed_qpos_adr + 7] = weed_qpos

    def _randomize_robot_qpos(self) -> None:
        noise = float(self.dr_config["robot_qpos_noise"])
        qpos = self._nominal_arm_qpos + self.np_random.uniform(-noise, noise, 7)
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        self.data.qpos[self.arm_qpos_adr] = np.clip(qpos, lower + 0.01, upper - 0.01)

    def close(self) -> None:
        self._physics = None


def run_env_checks(
    xml_path: str | Path | None = None,
    names: SceneNames | None = None,
) -> None:
    """Run both Gymnasium and Stable-Baselines3 environment checks."""
    from gymnasium.utils.env_checker import check_env as gym_check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    env = WeedPickEnv(xml_path=xml_path, names=names)
    try:
        gym_check_env(env, skip_render_check=True)
        sb3_check_env(env, warn=True)
    finally:
        env.close()
