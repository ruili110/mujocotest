# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dm-control>=1.0.43",
#     "jax[cuda12]==0.8.2",
#     "mujoco==3.11.0",
#     "mujoco-mjx[warp]==3.11.0",
#     "playground==0.2.0",
#     "wandb>=0.22.0",
# ]
# ///
"""GPU-parallel MJX-Warp environment and PPO trainer for weed picking.

Run:
    uv run --script train_weed_mjx_warp.py --run-name weed-pick-01

The environment keeps the original 7D action layout:
    [dx, dy, dz, dRx, dRy, dRz, gripper]
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
import numpy as np
import wandb

from new2 import PANDA_HOME_QPOS, build_scene
from trajectory_dataset import MjxTrainingTrajectoryCollector


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = ROOT.parent / "run"
DEFAULT_RUN_NAME = "ppo_weed_pick_mjx_warp"

ARM_JOINT_NAMES = tuple(f"panda/joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"panda/actuator{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = (
    "panda/umi_gripper/left_finger_joint",
    "panda/umi_gripper/right_finger_joint",
)
GRIPPER_ACTUATOR_NAME = "panda/umi_gripper/fingers_actuator"
EE_SITE_NAME = "panda/umi_gripper/ee_site"
WEED_BODY_NAME = "plant"
WEED_JOINT_NAME = "plant_freejoint"
WEED_GEOM_NAME = "plant_collision"
LEFT_PAD_GEOM_NAME = "panda/umi_gripper/left_finger_pad"
RIGHT_PAD_GEOM_NAME = "panda/umi_gripper/right_finger_pad"
LEFT_CONTACT_SENSOR = "left_weed_contact"
RIGHT_CONTACT_SENSOR = "right_weed_contact"

DEFAULT_GRIPPER_OPENING = 0.04
FINAL_WEED_POSITION = (0.300, 0.345, 0.375)
MIDDLE_TARGET_POSITION = (0.340, 0.180, 0.485)


def default_config(num_envs: int = 1024) -> config_dict.ConfigDict:
    """Returns the small set of environment settings used for training."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=200,
        action_repeat=1,
        pos_scale=0.02,
        rot_scale=0.1,
        ik_damping=0.01,
        ik_max_update_norm=0.2,
        target_tolerance=0.03,
        success_hold_steps=3,
        robot_qpos_noise=0.01,
        weed_lateral_offsets=(-0.09, 0.0, 0.09),
        weed_mass_scale_range=(0.80, 1.20),
        weed_friction_scale_range=(0.70, 1.30),
        impl="warp",
        naconmax=32 * num_envs,
        naccdmax=32 * num_envs,
        njmax=256,
    )


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    weed_body_id: int,
    weed_geom_id: int,
    mass_scale_range: tuple[float, float],
    friction_scale_range: tuple[float, float],
) -> tuple[mjx.Model, mjx.Model]:
    """Randomizes weed mass and friction independently for every environment."""

    @jax.vmap
    def randomize(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        mass_rng, friction_rng = jax.random.split(rng)
        mass_scale = jax.random.uniform(
            mass_rng,
            minval=mass_scale_range[0],
            maxval=mass_scale_range[1],
        )
        friction_scale = jax.random.uniform(
            friction_rng,
            minval=friction_scale_range[0],
            maxval=friction_scale_range[1],
        )
        body_mass = model.body_mass.at[weed_body_id].set(
            model.body_mass[weed_body_id] * mass_scale
        )
        geom_friction = model.geom_friction.at[weed_geom_id].set(
            model.geom_friction[weed_geom_id] * friction_scale
        )
        return body_mass, geom_friction

    body_mass, geom_friction = randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {"body_mass": 0, "geom_friction": 0}
    )
    model = model.tree_replace(
        {"body_mass": body_mass, "geom_friction": geom_friction}
    )
    return model, in_axes


class WeedPickMjxWarpEnv(mjx_env.MjxEnv):
    """Functional, JIT-safe weed-picking environment for MJX-Warp."""

    def __init__(self, config: config_dict.ConfigDict):
        super().__init__(config)

        scene = build_scene(export_xml=False)
        self._add_contact_sensors(scene)

        # MJX-Warp does not use MuJoCo's noslip pass. Euler is the conventional
        # simple integrator for massively parallel RL environments.
        scene.option.integrator = "Euler"
        scene.option.noslip_iterations = 0
        scene.option.timestep = config.sim_dt

        self._mj_model = mujoco.MjModel.from_xml_string(
            scene.to_xml_string(), assets=scene.get_assets()
        )
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)

        self._arm_joint_ids = np.array(
            [self._mj_model.joint(name).id for name in ARM_JOINT_NAMES]
        )
        self._arm_qpos_ids = self._mj_model.jnt_qposadr[
            self._arm_joint_ids
        ].astype(np.int32)
        self._arm_qvel_ids = self._mj_model.jnt_dofadr[
            self._arm_joint_ids
        ].astype(np.int32)
        self._arm_actuator_ids = np.array(
            [self._mj_model.actuator(name).id for name in ARM_ACTUATOR_NAMES]
        )
        self._finger_joint_ids = np.array(
            [self._mj_model.joint(name).id for name in FINGER_JOINT_NAMES]
        )
        self._finger_qpos_ids = self._mj_model.jnt_qposadr[
            self._finger_joint_ids
        ].astype(np.int32)
        self._gripper_actuator_id = self._mj_model.actuator(
            GRIPPER_ACTUATOR_NAME
        ).id
        self._ee_site_id = self._mj_model.site(EE_SITE_NAME).id
        self._ee_site_body_id = int(self._mj_model.site_bodyid[self._ee_site_id])
        self._ee_site_quat = jp.asarray(
            self._mj_model.site_quat[self._ee_site_id]
        )
        self._weed_body_id = self._mj_model.body(WEED_BODY_NAME).id
        self._weed_geom_id = self._mj_model.geom(WEED_GEOM_NAME).id
        weed_joint_id = self._mj_model.joint(WEED_JOINT_NAME).id
        self._weed_qpos_adr = int(self._mj_model.jnt_qposadr[weed_joint_id])

        self._left_sensor_adr = self._sensor_address(LEFT_CONTACT_SENSOR)
        self._right_sensor_adr = self._sensor_address(RIGHT_CONTACT_SENSOR)
        self._arm_lower = jp.asarray(
            self._mj_model.jnt_range[self._arm_joint_ids, 0]
        )
        self._arm_upper = jp.asarray(
            self._mj_model.jnt_range[self._arm_joint_ids, 1]
        )
        self._finger_lower = jp.asarray(
            self._mj_model.jnt_range[self._finger_joint_ids, 0]
        )
        self._finger_upper = jp.asarray(
            self._mj_model.jnt_range[self._finger_joint_ids, 1]
        )
        self._home_arm = jp.asarray(PANDA_HOME_QPOS)
        self._default_qpos = jp.asarray(self._mj_model.qpos0)
        self._final_weed_position = jp.asarray(FINAL_WEED_POSITION)

        cpu_data = mujoco.MjData(self._mj_model)
        cpu_data.qpos[self._arm_qpos_ids] = PANDA_HOME_QPOS
        mujoco.mj_forward(self._mj_model, cpu_data)
        self._weed_rest_height = float(cpu_data.geom_xpos[self._weed_geom_id, 2])

    @staticmethod
    def _add_contact_sensors(scene: Any) -> None:
        weed_geom = scene.find("geom", WEED_GEOM_NAME)
        left_pad_geom = scene.find("geom", LEFT_PAD_GEOM_NAME)
        right_pad_geom = scene.find("geom", RIGHT_PAD_GEOM_NAME)
        sensor_kwargs = {
            "geom2": weed_geom,
            "reduce": "netforce",
            "num": 1,
            "data": "found force normal",
        }
        scene.sensor.add(
            "contact",
            name=LEFT_CONTACT_SENSOR,
            geom1=left_pad_geom,
            **sensor_kwargs,
        )
        scene.sensor.add(
            "contact",
            name=RIGHT_CONTACT_SENSOR,
            geom1=right_pad_geom,
            **sensor_kwargs,
        )

    def _sensor_address(self, name: str) -> int:
        sensor_id = self._mj_model.sensor(name).id
        return int(self._mj_model.sensor_adr[sensor_id])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, weed_rng, robot_rng = jax.random.split(rng, 3)
        lateral_offset = jax.random.choice(
            weed_rng, jp.asarray(self._config.weed_lateral_offsets)
        )
        arm_qpos = self._home_arm + jax.random.uniform(
            robot_rng,
            shape=(7,),
            minval=-self._config.robot_qpos_noise,
            maxval=self._config.robot_qpos_noise,
        )
        arm_qpos = jp.clip(arm_qpos, self._arm_lower + 0.01, self._arm_upper - 0.01)

        finger_qpos = self._gripper_opening_to_joint_positions(
            DEFAULT_GRIPPER_OPENING
        )
        qpos = self._default_qpos
        qpos = qpos.at[self._arm_qpos_ids].set(arm_qpos)
        qpos = qpos.at[self._finger_qpos_ids].set(finger_qpos)
        qpos = qpos.at[self._weed_qpos_adr + 1].add(lateral_offset)

        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._arm_actuator_ids].set(arm_qpos)
        ctrl = ctrl.at[self._gripper_actuator_id].set(
            self._gripper_opening_to_ctrl(DEFAULT_GRIPPER_OPENING)
        )
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=jp.zeros(self._mj_model.nv),
            ctrl=ctrl,
            impl=self._config.impl,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)
        middle_target_position = jp.asarray(MIDDLE_TARGET_POSITION)

        info = {
            "rng": rng,
            "prev_action": jp.zeros(7),
            "success_hold_count": jp.array(0, dtype=jp.int32),
            "grasp_once": jp.array(False),
            "weed_clears_pot_once": jp.array(False),
            "target_reached_once": jp.array(False),
            "middle_target_position": middle_target_position,
        }
        metrics = self._empty_metrics()
        contact = self._contact_state(data)
        obs = self._get_obs(data, info["prev_action"], contact["is_grasped"])
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(
        self, state: mjx_env.State, action: jax.Array
    ) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        arm_target = self._cartesian_action_to_joint_target(state.data, action[:6])
        gripper_target = self._normalized_gripper_target(action[6])
        ctrl = state.data.ctrl
        ctrl = ctrl.at[self._arm_actuator_ids].set(arm_target)
        ctrl = ctrl.at[self._gripper_actuator_id].set(gripper_target)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        contact = self._contact_state(data)
        info = dict(state.info)
        task = self._task_state(data, info["middle_target_position"])
        success_hold_count = jp.where(
            task["target_reached"],
            info["success_hold_count"] + 1,
            0,
        )
        success = success_hold_count >= self._config.success_hold_steps
        info.update(
            prev_action=action,
            success_hold_count=success_hold_count,
            grasp_once=info["grasp_once"] | contact["is_grasped"],
            weed_clears_pot_once=(
                info["weed_clears_pot_once"] | task["weed_clears_pot"]
            ),
            target_reached_once=(
                info["target_reached_once"] | task["target_reached"]
            ),
        )

        failure = jp.any(~jp.isfinite(data.qpos)) | (task["weed_pos"][2] < -0.1)
        reward = self._reward(task, success, failure)
        done = failure.astype(float)
        obs = self._get_obs(data, action, contact["is_grasped"])
        metrics = dict(state.metrics)
        metrics.update(
            success=success.astype(float),
            is_grasped=contact["is_grasped"].astype(float),
            grasp_once=info["grasp_once"].astype(float),
            weed_clears_pot_once=info["weed_clears_pot_once"].astype(float),
            target_reached_once=info["target_reached_once"].astype(float),
            reach_distance=task["reach_distance"],
            target_distance=task["target_distance"],
            lift_amount=task["lift_amount"],
            left_grasp_force=contact["left_force"],
            right_grasp_force=contact["right_force"],
        )
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _cartesian_action_to_joint_target(
        self, data: mjx.Data, action: jax.Array
    ) -> jax.Array:
        """One batched damped-least-squares IK update."""
        rotation = action[3:]
        rotation_norm = jp.linalg.norm(rotation)
        rotation = rotation / jp.maximum(rotation_norm, 1.0)
        twist = jp.concatenate(
            (action[:3] * self._config.pos_scale, rotation * self._config.rot_scale)
        )

        axes = data.xaxis[self._arm_joint_ids]
        anchors = data.xanchor[self._arm_joint_ids]
        ee_position = data.site_xpos[self._ee_site_id]
        linear_jacobian = jax.vmap(jp.cross)(axes, ee_position - anchors).T
        angular_jacobian = axes.T
        jacobian = jp.concatenate((linear_jacobian, angular_jacobian), axis=0)
        regularizer = self._config.ik_damping**2 * jp.eye(6)
        joint_delta = jacobian.T @ jp.linalg.solve(
            jacobian @ jacobian.T + regularizer, twist
        )
        delta_norm = jp.linalg.norm(joint_delta)
        joint_delta *= jp.minimum(
            1.0, self._config.ik_max_update_norm / jp.maximum(delta_norm, 1e-8)
        )
        target = data.qpos[self._arm_qpos_ids] + joint_delta
        return jp.clip(target, self._arm_lower, self._arm_upper)

    def _contact_state(self, data: mjx.Data) -> dict[str, jax.Array]:
        left = data.sensordata[self._left_sensor_adr : self._left_sensor_adr + 7]
        right = data.sensordata[
            self._right_sensor_adr : self._right_sensor_adr + 7
        ]
        left_force_vector = left[1:4]
        right_force_vector = right[1:4]
        left_force = jp.linalg.norm(left_force_vector)
        right_force = jp.linalg.norm(right_force_vector)

        # Contact is only observed/logged; it does not gate reward or success.
        left_contact = left[0] > 0
        right_contact = right[0] > 0
        return {
            "is_grasped": left_contact & right_contact,
            "left_force": left_force,
            "right_force": right_force,
        }

    def _task_state(
        self,
        data: mjx.Data,
        middle_target_position: jax.Array,
    ) -> dict[str, jax.Array]:
        ee_pos = data.site_xpos[self._ee_site_id]
        weed_pos = data.geom_xpos[self._weed_geom_id]
        reach_distance = jp.linalg.norm(weed_pos - ee_pos)
        middle_target_distance = jp.linalg.norm(
            weed_pos - middle_target_position
        )
        target_distance = jp.linalg.norm(weed_pos - self._final_weed_position)
        lift_amount = weed_pos[2] - self._weed_rest_height
        target_reached = target_distance <= self._config.target_tolerance
        weed_clears_pot = lift_amount >= 0.08
        return {
            "weed_pos": weed_pos,
            "reach_distance": reach_distance,
            "middle_target_distance": middle_target_distance,
            "target_distance": target_distance,
            "lift_amount": lift_amount,
            "target_reached": target_reached,
            "weed_clears_pot": weed_clears_pot,
        }

    @staticmethod
    def _reward(
        task: dict[str, jax.Array],
        success: jax.Array,
        failure: jax.Array,
    ) -> jax.Array:
        reaching_reward = 1.0 - jp.tanh(5.0 * task["reach_distance"])
        middle_target_reward = 1.0 - jp.tanh(
            2.5 * task["middle_target_distance"]
        )
        target_reward = 1.0 - jp.tanh(2.5 * task["target_distance"])
        dense_reward = (
            reaching_reward
            + 2.0 * middle_target_reward
            + 3.0 * target_reward
        )
        dense_reward = jp.where(success, 6.0, dense_reward)
        return jp.where(failure, -1.0, dense_reward / 6.0)

    def _get_obs(
        self, data: mjx.Data, previous_action: jax.Array, is_grasped: jax.Array
    ) -> jax.Array:
        arm_qpos = data.qpos[self._arm_qpos_ids]
        arm_qvel = data.qvel[self._arm_qvel_ids]
        ee_pos = data.site_xpos[self._ee_site_id]
        ee_quat = mjx_math.quat_mul(
            data.xquat[self._ee_site_body_id], self._ee_site_quat
        )
        ee_quat = jp.where(ee_quat[0] < 0.0, -ee_quat, ee_quat)
        weed_pos = data.geom_xpos[self._weed_geom_id]
        weed_quat = data.xquat[self._weed_body_id]
        weed_quat = jp.where(weed_quat[0] < 0.0, -weed_quat, weed_quat)
        gripper_opening = jp.sum(
            self._finger_upper
            - jp.clip(
                data.qpos[self._finger_qpos_ids],
                self._finger_lower,
                self._finger_upper,
            )
        )
        return jp.concatenate(
            (
                arm_qpos,
                arm_qvel,
                ee_pos,
                ee_quat,
                jp.asarray([gripper_opening]),
                jp.asarray([is_grasped], dtype=float),
                weed_quat,
                weed_pos - ee_pos,
                self._final_weed_position - weed_pos,
                previous_action,
            )
        )

    def _gripper_opening_to_joint_positions(self, opening: float) -> jax.Array:
        max_opening = jp.sum(self._finger_upper - self._finger_lower)
        fraction = jp.clip(opening / max_opening, 0.0, 1.0)
        return self._finger_upper - fraction * (
            self._finger_upper - self._finger_lower
        )

    def _gripper_opening_to_ctrl(self, opening: float) -> jax.Array:
        ctrl_min, ctrl_max = self._mj_model.actuator_ctrlrange[
            self._gripper_actuator_id
        ]
        max_opening = jp.sum(self._finger_upper - self._finger_lower)
        fraction = jp.clip(opening / max_opening, 0.0, 1.0)
        return ctrl_max - fraction * (ctrl_max - ctrl_min)

    def _normalized_gripper_target(self, command: jax.Array) -> jax.Array:
        ctrl_min, ctrl_max = self._mj_model.actuator_ctrlrange[
            self._gripper_actuator_id
        ]
        opening_fraction = 0.5 * (command + 1.0)
        return ctrl_max - opening_fraction * (ctrl_max - ctrl_min)

    @staticmethod
    def _empty_metrics() -> dict[str, jax.Array]:
        names = (
            "success",
            "is_grasped",
            "grasp_once",
            "weed_clears_pot_once",
            "target_reached_once",
            "reach_distance",
            "target_distance",
            "lift_amount",
            "left_grasp_force",
            "right_grasp_force",
        )
        return {name: jp.array(0.0) for name in names}

    @property
    def action_size(self) -> int:
        return 7

    @property
    def xml_path(self) -> str:
        return str(ROOT / "merged_full_scene.xml")

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weed-picking PPO on MJX-Warp.")
    parser.add_argument("--timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--num-envs", type=int, default=3072)
    parser.add_argument("--num-evals", type=int, default=101)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--wandb-project", default="weed-picking")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "MJX-Warp PPO needs an NVIDIA GPU; JAX currently reports "
            f"{jax.default_backend()!r}."
        )

    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one directory name")
    num_minibatches = 8
    if args.num_envs % num_minibatches:
        raise ValueError("--num-envs must be divisible by 8")
    batch_size = args.num_envs // num_minibatches

    output_dir = (DEFAULT_RUN_ROOT / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config = default_config(args.num_envs)
    env = WeedPickMjxWarpEnv(env_config)
    trajectory_env = WeedPickMjxWarpEnv(default_config(args.num_eval_envs))
    trajectory_collector = MjxTrainingTrajectoryCollector(
        trajectory_env,
        output_dir / "successful_trajectories",
        num_episodes=args.num_eval_envs,
        seed=args.seed,
    )
    randomization_fn = functools.partial(
        domain_randomize,
        weed_body_id=env.mj_model.body(WEED_BODY_NAME).id,
        weed_geom_id=env.mj_model.geom(WEED_GEOM_NAME).id,
        mass_scale_range=tuple(env_config.weed_mass_scale_range),
        friction_scale_range=tuple(env_config.weed_friction_scale_range),
    )
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )

    def progress(step: int, metrics: dict[str, Any]) -> None:
        reward = float(metrics.get("eval/episode_reward", 0.0))
        print(f"step={step} eval_reward={reward:.3f}")
        scalar_metrics = {}
        for name, value in metrics.items():
            value = np.asarray(jax.device_get(value))
            if value.size == 1:
                scalar_metrics[name] = float(value.item())
        run.log(scalar_metrics, step=step, commit=True)

    run_config = {
        "backend": "mjx-warp",
        "algorithm": "brax-ppo",
        "run_name": args.run_name,
        "timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "num_evals": args.num_evals,
        "num_eval_envs": args.num_eval_envs,
        "episode_length": env_config.episode_length,
        "batch_size": batch_size,
        "num_minibatches": num_minibatches,
        "num_updates_per_batch": 4,
        "unroll_length": 10,
        "discounting": 0.99,
        "learning_rate": 3e-4,
        "entropy_cost": 5e-3,
        "weed_mass_scale_range": list(env_config.weed_mass_scale_range),
        "weed_friction_scale_range": list(env_config.weed_friction_scale_range),
        "seed": args.seed,
    }
    with wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=run_config,
    ) as run:
        print(
            f"Training on {jax.devices()[0]} with {args.num_envs} environments. "
            f"Checkpoints: {output_dir}"
        )
        ppo.train(
            environment=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            num_timesteps=args.timesteps,
            num_evals=args.num_evals,
            episode_length=env_config.episode_length,
            action_repeat=env_config.action_repeat,
            num_envs=args.num_envs,
            num_eval_envs=args.num_eval_envs,
            randomization_fn=randomization_fn,
            max_devices_per_host=1,
            unroll_length=10,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=4,
            discounting=0.99,
            learning_rate=3e-4,
            entropy_cost=5e-3,
            reward_scaling=0.1,
            clipping_epsilon=0.2,
            gae_lambda=0.95,
            normalize_observations=True,
            max_grad_norm=1.0,
            network_factory=network_factory,
            seed=args.seed,
            progress_fn=progress,
            policy_params_fn=trajectory_collector,
            save_checkpoint_path=str(output_dir),
        )


if __name__ == "__main__":
    train(parse_args())
