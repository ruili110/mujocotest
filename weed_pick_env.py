"""Gymnasium MuJoCo environment for a Franka weed pick task.

Scope of this version:
- proprioception + relative weed position
- 6D delta end-effector action + continuous gripper-width action
- damped least-squares differential IK
- normalized ManiSkill-style staged dense reward
- with domain randomization


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
from dm_control.utils import inverse_kinematics

from new2 import PANDA_HOME_QPOS, build_scene


MIN_GRASP_FORCE = 0.5
MAX_GRASP_FORCE_ANGLE_DEG = 85.0


@dataclass(frozen=True)
class SceneNames:
    ee_site: str = "panda/umi_gripper/ee_site"
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


@dataclass(frozen=True)
class GraspQuality:
    left_contact: bool
    right_contact: bool
    left_force: float
    right_force: float
    left_force_angle: float
    right_force_angle: float

    @property
    def left_valid(self) -> bool:
        return bool(
            self.left_contact
            and self.left_force >= MIN_GRASP_FORCE
            and self.left_force_angle <= MAX_GRASP_FORCE_ANGLE_DEG
        )

    @property
    def right_valid(self) -> bool:
        return bool(
            self.right_contact
            and self.right_force >= MIN_GRASP_FORCE
            and self.right_force_angle <= MAX_GRASP_FORCE_ANGLE_DEG
        )

    @property
    def is_grasped(self) -> bool:
        return bool(self.left_valid and self.right_valid)


DEFAULT_DR_CONFIG: dict[str, Any] = {
    "weed_lateral_offsets": (-0.09, 0.0, 0.09),
    "robot_qpos_noise": 0.01,
    "action_noise_std": 0.0,
    # Kept deliberately narrow for the first working version.
    "weed_mass_scale_range": (0.95, 1.05),
    "weed_friction_scale_range": (0.9, 1.1),
}


HOME_ARM = PANDA_HOME_QPOS.copy()
DEFAULT_GRIPPER_OPENING = 0.04
LIFT_REWARD_SCALE = 10.0
LIFT_REWARD_WEIGHT = 2.0
TARGET_REWARD_WEIGHT = 3.0
MAX_DENSE_REWARD = 6.0
STATIC_REWARD_SPEED_SCALE = 5.0
JOINT_LIMIT_MARGIN = 0.0
IK_MAX_STEPS = 100
IK_POSITION_TOLERANCE = 1e-4
IK_MAX_UPDATE_NORM = 0.2


class WeedPickEnv(gym.Env[np.ndarray, np.ndarray]):
    """Pick-only Franka environment.

    The task is to grasp the weed and move it to the configured final position.

    Action layout, all normalized to [-1, 1]:
        [dx, dy, dz, dRx, dRy, dRz, gripper_width]

    The last component is continuous. -1 closes the gripper and +1 opens it.
    """

    metadata = {"render_modes": []}

    OBS_COMPONENTS = {
        "arm_qpos": 7,
        "arm_qvel": 7,
        "ee_pos": 3,
        "ee_quat": 4,
        "gripper_width": 1,
        "is_grasped": 1,
        "weed_quat": 4,
        "weed_rel_ee": 3,
        "final_rel_weed": 3,
        "last_action": 7,
    } ## Is-grasped is important for the agent to fully differentiate itself from when it is just close to the weed to when it has grasped the weed
    OBS_DIM = sum(OBS_COMPONENTS.values())

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        names: SceneNames | None = None,
        dr_config: dict[str, Any] | None = None,
        home_arm: np.ndarray = HOME_ARM,
        max_episode_steps: int = 200,
        frame_skip: int = 10,
        pos_scale: float = 0.02,
        rot_scale: float = 0.1,
        ik_damping: float = 0.01,
        clearance_height: float = 0.08,
        safe_lift_height: float = 0.08,
        final_weed_position: tuple[float, float, float] = (0.50, 0.335, 0.54),
        target_tolerance: float = 0.03,
        success_hold_steps: int = 5,
    ) -> None:
        super().__init__()

        self.names = names or SceneNames()
        if xml_path is None:
            scene = build_scene(export_xml=False)
            self._physics = mjcf.Physics.from_mjcf_model(scene)
        else:
            scene_path = Path(xml_path).expanduser().resolve()
            if not scene_path.is_file():
                raise FileNotFoundError(f"MJCF file not found: {scene_path}")
            self._physics = mjcf.Physics.from_xml_path(str(scene_path))
        self.model = self._physics.model.ptr
        self.data = self._physics.data.ptr
        mujoco.mj_forward(self.model, self.data)

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.ik_damping = float(ik_damping)
        self.clearance_height = float(clearance_height)
        if self.clearance_height <= 0.0:
            raise ValueError("clearance_height must be positive.")
        self.safe_lift_height = float(safe_lift_height)
        if self.safe_lift_height < self.clearance_height:
            raise ValueError(
                "safe_lift_height must be greater than or equal to clearance_height."
            )
        self._final_weed_position = np.asarray(
            final_weed_position, dtype=np.float64
        ).copy()
        self.target_tolerance = float(target_tolerance)

        self.success_hold_steps = int(success_hold_steps)

        self.dr_config = deepcopy(DEFAULT_DR_CONFIG)
        if dr_config is not None:
            self.dr_config.update(deepcopy(dr_config))

        self._cache_model_ids()
        self._nominal_arm_qpos = np.asarray(home_arm, dtype=np.float64).copy()
        self._validate_home_pose()
        self._default_weed_qpos = self.model.qpos0[
            self.weed_qpos_adr : self.weed_qpos_adr + 7
        ].copy()
        self._weed_rest_height = float(self._weed_position()[2])
        self._default_body_mass = self.model.body_mass.copy()
        self._default_geom_friction = self.model.geom_friction.copy()

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
        self._success_hold_count = 0
        self._grasp_once = False
        self._left_contact_once = False
        self._right_contact_once = False
        self._weed_clears_pot_once = False
        self._target_reached_once = False

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._prev_action.fill(0.0)
        self._success_hold_count = 0
        self._grasp_once = False
        self._left_contact_once = False
        self._right_contact_once = False
        self._weed_clears_pot_once = False
        self._target_reached_once = False

        self._restore_nominal_model_parameters()
        self._apply_domain_randomization()
        self._randomize_weed_pose()
        self._randomize_robot_qpos()
        self.data.qpos[self.finger_qpos_adr] = (
            self._gripper_opening_to_joint_positions(DEFAULT_GRIPPER_OPENING)
        ) ##qpos is for here the fingers currently are, and ctrl is for where we want the fingers to go
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_adr]
        self.data.ctrl[self.gripper_actuator_id] = (
            self._gripper_opening_to_ctrl_target(DEFAULT_GRIPPER_OPENING)
        )
        mujoco.mj_forward(self.model, self.data)

        info = self._get_info()
        return self._get_obs(is_grasped=info["is_grasped"]), info

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

        self._prev_action = executed_action.copy()
        self._step_count += 1

        info = self._get_info()
        reward = self._compute_reward(info)
        # Keep successful episodes on the fixed horizon so reaching the target
        # earlier produces more successful reward steps.
        terminated = bool(info["failure"])
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        return (
            self._get_obs(is_grasped=info["is_grasped"]),
            reward,
            terminated,
            truncated,
            info,
        )

    def _cache_model_ids(self) -> None:
        self.ee_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, self.names.ee_site
        )
        self.weed_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, self.names.weed_body
        )
        self.weed_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT, self.names.weed_joint
        )

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
        rotation_action = np.asarray(ee_delta[3:], dtype=np.float64)
        rotation_norm = float(np.linalg.norm(rotation_action))
        if rotation_norm > 1.0:
            rotation_action = rotation_action / rotation_norm
        drot = rotation_action * self.rot_scale

        target_position = self.data.site_xpos[self.ee_site_id].copy() + dpos
        current_quaternion = self._site_xquat(self.ee_site_id)
        target_quaternion = self._apply_rotation_delta(current_quaternion, drot)

        q_target = self._solve_ik(target_position, target_quaternion)
        if q_target is None:
            return self.data.ctrl[self.arm_actuator_ids].copy()
        return q_target

    @staticmethod
    def _apply_rotation_delta(
        current_quaternion: np.ndarray,
        rotation_delta: np.ndarray,
    ) -> np.ndarray:
        """Apply a root-aligned rotation-vector delta to an orientation."""
        rotation_angle = float(np.linalg.norm(rotation_delta))
        if rotation_angle <= 1e-12:
            return current_quaternion.copy()

        delta_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(
            delta_quaternion,
            rotation_delta / rotation_angle,
            rotation_angle,
        )
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(
            target_quaternion,
            delta_quaternion,
            current_quaternion,
        )
        return target_quaternion

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> np.ndarray | None:
        """Use standard full-pose IK and reject infeasible solutions."""
        result = inverse_kinematics.qpos_from_site_pose(
            self._physics,
            self.names.ee_site,
            target_pos=target_position,
            target_quat=target_quaternion,
            joint_names=self.names.arm_joints,
            tol=IK_POSITION_TOLERANCE,
            regularization_strength=self.ik_damping,
            max_update_norm=IK_MAX_UPDATE_NORM,
            max_steps=IK_MAX_STEPS,
            inplace=False,
        )
        if not result.success:
            return None

        q_target = result.qpos[self.arm_qpos_adr].copy()
        lower = (
            self.model.jnt_range[self.arm_joint_ids, 0] + JOINT_LIMIT_MARGIN
        )
        upper = (
            self.model.jnt_range[self.arm_joint_ids, 1] - JOINT_LIMIT_MARGIN
        )
        target_is_feasible = bool(
            np.all(np.isfinite(q_target))
            and np.all(q_target >= lower)
            and np.all(q_target <= upper)
        )
        return q_target if target_is_feasible else None

    def _normalized_to_gripper_target(self, command: float) -> float:
        """Map -1 (closed) and +1 (open) to the actuator control range."""
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        opening_fraction = 0.5 * (
            float(np.clip(command, -1.0, 1.0)) + 1.0
        )
        return float(ctrl_max - opening_fraction * (ctrl_max - ctrl_min)) ##it becomes inverted as in theactruator for the gripper, a higher command closes the target and lower command opens it.

    def _gripper_opening_to_joint_positions(self, opening: float) -> np.ndarray:
        lower = self.model.jnt_range[self.finger_joint_ids, 0]
        upper = self.model.jnt_range[self.finger_joint_ids, 1]
        max_opening = float(np.sum(upper - lower))
        opening_fraction = float(np.clip(opening / max_opening, 0.0, 1.0))
        return upper - opening_fraction * (upper - lower)

    def _gripper_opening_to_ctrl_target(self, opening: float) -> float:
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        lower = self.model.jnt_range[self.finger_joint_ids, 0]
        upper = self.model.jnt_range[self.finger_joint_ids, 1]
        max_opening = float(np.sum(upper - lower))
        opening_fraction = float(np.clip(opening / max_opening, 0.0, 1.0))
        return float(ctrl_max - opening_fraction * (ctrl_max - ctrl_min))

    def _current_gripper_opening(self) -> float:
        lower = self.model.jnt_range[self.finger_joint_ids, 0]
        upper = self.model.jnt_range[self.finger_joint_ids, 1]
        qpos = np.clip(self.data.qpos[self.finger_qpos_adr], lower, upper)
        return float(np.sum(upper - qpos))

    ##command -1 → actuator 0.05 → finger qpos 0.05 → opening 0.00 → closed
    ##command +1 → actuator 0.00 → finger qpos 0.00 → opening 0.10 → open


    def _get_obs(self, *, is_grasped: bool | None = None) -> np.ndarray:
        arm_qpos = self.data.qpos[self.arm_qpos_adr].copy()
        arm_qvel = self.data.qvel[self.arm_dof_adr].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_quat = self._site_xquat(self.ee_site_id)
        if ee_quat[0] < 0.0:
            ee_quat = -ee_quat

        weed_pos = self._weed_position()
        weed_quat = self._weed_quaternion()
        final_weed_pos = self._final_weed_position
        if is_grasped is None:
            is_grasped = self._is_grasping()
        obs = np.concatenate(
            (
                arm_qpos,
                arm_qvel,
                ee_pos,
                ee_quat,
                np.array([self._current_gripper_opening()]),
                np.array([float(is_grasped)]),
                weed_quat,
                weed_pos - ee_pos,
                final_weed_pos - weed_pos,
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
        return np.mean(self.data.geom_xpos[geom_ids], axis=0)

    def _weed_quaternion(self) -> np.ndarray:
        """Return the weed's world-frame orientation as a canonical wxyz quaternion."""
        quaternion = self.data.xquat[self.weed_body_id].copy()
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        return quaternion

    def _site_xquat(self, site_id: int) -> np.ndarray:
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return quat

    @staticmethod
    def _force_angle_degrees(force: np.ndarray, direction: np.ndarray) -> float:
        force_norm = float(np.linalg.norm(force))
        direction_norm = float(np.linalg.norm(direction))
        if force_norm <= 1e-12 or direction_norm <= 1e-12:
            return 180.0
        cosine = float(
            np.dot(force, direction) / (force_norm * direction_norm)
        )
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _grasp_quality(self) -> GraspQuality:
        left_force_vector = np.zeros(3, dtype=np.float64)
        right_force_vector = np.zeros(3, dtype=np.float64)
        left_contact = False
        right_contact = False

        left_pad_position = np.mean(
            self.data.geom_xpos[list(self.left_pad_geom_ids)], axis=0
        )
        right_pad_position = np.mean(
            self.data.geom_xpos[list(self.right_pad_geom_ids)], axis=0
        )
        left_closing_direction = right_pad_position - left_pad_position
        right_closing_direction = -left_closing_direction

        for contact_index, contact in enumerate(
            self.data.contact[: self.data.ncon]
        ):
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            pair = {geom1, geom2}
            if not pair & self.weed_geom_ids:
                continue

            if pair & self.left_pad_geom_ids:
                pad_geom_ids = self.left_pad_geom_ids
                force_vector = left_force_vector
                left_contact = True
            elif pair & self.right_pad_geom_ids:
                pad_geom_ids = self.right_pad_geom_ids
                force_vector = right_force_vector
                right_contact = True
            else:
                continue

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                self.model,
                self.data,
                contact_index,
                contact_force,
            )
            normal_force = max(float(contact_force[0]), 0.0)
            contact_normal = np.asarray(contact.frame).reshape(3, 3)[0]
            if geom1 not in pad_geom_ids:
                contact_normal = -contact_normal
            force_vector += normal_force * contact_normal

        left_force = float(np.linalg.norm(left_force_vector))
        right_force = float(np.linalg.norm(right_force_vector))
        return GraspQuality(
            left_contact=left_contact,
            right_contact=right_contact,
            left_force=left_force,
            right_force=right_force,
            left_force_angle=self._force_angle_degrees(
                left_force_vector,
                left_closing_direction,
            ),
            right_force_angle=self._force_angle_degrees(
                right_force_vector,
                right_closing_direction,
            ),
        )

    def _finger_contact_state(self) -> tuple[bool, bool]:
        quality = self._grasp_quality()
        return quality.left_valid, quality.right_valid

    def _is_grasping(self) -> bool:
        return self._grasp_quality().is_grasped

    def _get_info(self) -> dict[str, Any]:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        weed_pos = self._weed_position()
        reach_distance = float(np.linalg.norm(weed_pos - ee_pos))
        final_weed_pos = self._final_weed_position
        grasp_quality = self._grasp_quality()
        target_distance = float(np.linalg.norm(weed_pos - final_weed_pos))
        robot_speed = float(
            np.linalg.norm(self.data.qvel[self.arm_dof_adr])
        )
        is_robot_static = bool(
            np.max(np.abs(self.data.qvel[self.arm_dof_adr])) <= 0.2
        )
        is_grasped = grasp_quality.is_grasped
        self._left_contact_once |= grasp_quality.left_contact
        self._right_contact_once |= grasp_quality.right_contact
        self._grasp_once |= is_grasped
        lift_amount = float(weed_pos[2] - self._weed_rest_height)
        target_reached = bool(
            target_distance <= self.target_tolerance
        )
        self._target_reached_once |= target_reached
        if target_reached and is_robot_static:
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0

        weed_clears_pot = bool(
            is_grasped and lift_amount >= self.clearance_height
        )
        self._weed_clears_pot_once |= weed_clears_pot
        success = bool(self._success_hold_count >= self.success_hold_steps)
        # Simple failure guard for obviously invalid episodes.
        failure = bool(
            not np.all(np.isfinite(self.data.qpos))
            or weed_pos[2] < -0.1
        )
        return {
            "success": success,
            "is_success": success,
            "failure": failure,
            "is_grasped": is_grasped,
            "left_contact": grasp_quality.left_contact,
            "right_contact": grasp_quality.right_contact,
            "left_grasp_valid": grasp_quality.left_valid,
            "right_grasp_valid": grasp_quality.right_valid,
            "left_grasp_force": grasp_quality.left_force,
            "right_grasp_force": grasp_quality.right_force,
            "left_grasp_force_angle": grasp_quality.left_force_angle,
            "right_grasp_force_angle": grasp_quality.right_force_angle,
            "grasp_once": self._grasp_once,
            "left_contact_once": self._left_contact_once,
            "right_contact_once": self._right_contact_once,
            "reach_distance": reach_distance,
            "final_weed_position": final_weed_pos.copy(),
            "target_distance": target_distance,
            "target_reached": target_reached,
            "robot_speed": robot_speed,
            "is_robot_static": is_robot_static,
            "lift_amount": lift_amount,
            "weed_clears_pot": weed_clears_pot,
            "weed_clears_pot_once": self._weed_clears_pot_once,
            "target_reached_once": self._target_reached_once,
            "success_hold_count": self._success_hold_count,
            "control_dt": self.control_dt,
        }

    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Normalized ManiSkill PickCube-style reward."""
        if info["failure"]:
            return -1.0

        reaching_reward = 1.0 - np.tanh(5.0 * info["reach_distance"])
        target_reward = 1.0 - np.tanh(2.5 * info["target_distance"])
        static_reward = 1.0 - np.tanh(
            STATIC_REWARD_SPEED_SCALE * info["robot_speed"]
        )
        is_grasped = float(info["is_grasped"])
        positive_lift = float(
            np.clip(info["lift_amount"], 0.0, self.safe_lift_height)
        )
        lift_reward = np.tanh(LIFT_REWARD_SCALE * positive_lift) / np.tanh(
            LIFT_REWARD_SCALE * self.safe_lift_height
        )

        dense_reward = float(reaching_reward) + is_grasped
      ##dense_reward += LIFT_REWARD_WEIGHT * float(lift_reward) * is_grasped
        dense_reward += (
            TARGET_REWARD_WEIGHT
            * float(target_reward)
   ##       * float(lift_reward)
            * is_grasped
        )
        dense_reward += float(static_reward) * float(info["target_reached"])

        if info["success"]:
            dense_reward = MAX_DENSE_REWARD

        return dense_reward / MAX_DENSE_REWARD

    def _restore_nominal_model_parameters(self) -> None:
        self.model.body_mass[:] = self._default_body_mass
        self.model.geom_friction[:] = self._default_geom_friction

    def _apply_domain_randomization(self) -> None:
        mass_scale = self.np_random.uniform(
            *self.dr_config["weed_mass_scale_range"]
        )
        self.model.body_mass[self.weed_body_id] = (
            self._default_body_mass[self.weed_body_id] * mass_scale
        )

        friction_scale = self.np_random.uniform(
            *self.dr_config["weed_friction_scale_range"]
        )
        for geom_id in self.weed_geom_ids:
            self.model.geom_friction[geom_id] = (
                self._default_geom_friction[geom_id] * friction_scale
            )

    def _randomize_weed_pose(self) -> None:
        lateral_offsets = np.asarray(
            self.dr_config["weed_lateral_offsets"], dtype=np.float64
        )
        if (
            lateral_offsets.shape != (3,)
        ):
            raise ValueError(
                "weed_lateral_offsets must contain three distinct finite values."
            )

        weed_qpos = self._default_weed_qpos.copy()
        weed_qpos[1] += float(self.np_random.choice(lateral_offsets))

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
