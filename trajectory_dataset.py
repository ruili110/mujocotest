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
"""Camera-free storage for successful absolute-pose trajectories."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:
    BaseCallback = object


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
}
OBS_DIM = sum(OBS_COMPONENTS.values())

DEFAULT_TASK_NAME = "mystery_mission"
IMAGE_KEY = "observation.image.realsense"
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
SIM_CAMERA_NAME = "panda/umi_gripper/input_camera"
POST_SUCCESS_SECONDS = 1.0
SIM_GRIPPER_CLOSED = 0.0
SIM_GRIPPER_OPEN = 0.10
ARUCO_CLOSED = 187.5
ARUCO_OPEN = 500.0


STATE_FEATURE_NAMES = (
    "aruco_dist",
    "optitrack_trans_x",
    "optitrack_trans_y",
    "optitrack_trans_z",
    "optitrack_rot_feat_0",
    "optitrack_rot_f1",
    "optitrack_rot_f2",
    "optitrack_rot_f3",
    "optitrack_rot_f4",
    "optitrack_rot_f5",
)

ACTION_FEATURE_NAMES = (
    "aruco_dist",
    "optitrack_trans_x",
    "optitrack_trans_y",
    "optitrack_trans_z",
    "optitrack_rot_f0",
    "optitrack_rot_f1",
    "optitrack_rot_f2",
    "optitrack_rot_f3",
    "optitrack_rot_f4",
    "optitrack_rot_f5",
)


def _component_slice(component_name: str) -> slice:
    start = 0
    for name, size in OBS_COMPONENTS.items():
        if name == component_name:
            return slice(start, start + size)
        start += size
    raise KeyError(f"Unknown observation component: {component_name}")


EE_POS_SLICE = _component_slice("ee_pos")
EE_QUAT_SLICE = _component_slice("ee_quat")
GRIPPER_WIDTH_SLICE = _component_slice("gripper_width")
WEED_QUAT_SLICE = _component_slice("weed_quat")


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert MuJoCo wxyz quaternions to rotation matrices."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("Cannot convert a zero-length quaternion.")

    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def gripper_width_to_aruco_distance(gripper_width: np.ndarray) -> np.ndarray:
    """Map simulated finger separation to the real dataset's ArUco scale."""
    opening_fraction = np.clip(
        (np.asarray(gripper_width) - SIM_GRIPPER_CLOSED)
        / (SIM_GRIPPER_OPEN - SIM_GRIPPER_CLOSED),
        0.0,
        1.0,
    )
    return ARUCO_CLOSED + opening_fraction * (ARUCO_OPEN - ARUCO_CLOSED)


def observation_to_absolute_state(observation: np.ndarray) -> np.ndarray:
    """Build the converter-compatible 10D absolute state from simulator state."""
    observation = np.asarray(observation)
    if observation.shape[-1] != OBS_DIM:
        raise ValueError(
            f"Expected observation dimension {OBS_DIM}, "
            f"got {observation.shape[-1]}."
        )

    simulated_aruco_dist = gripper_width_to_aruco_distance(
        observation[..., GRIPPER_WIDTH_SLICE]
    )
    simulated_optitrack_translation = observation[..., EE_POS_SLICE]
    simulated_optitrack_quaternion = observation[..., EE_QUAT_SLICE]
    inverse_matrix = np.swapaxes(
        _quaternion_to_matrix(simulated_optitrack_quaternion), -1, -2
    )
    rotation_feature = inverse_matrix.reshape(inverse_matrix.shape[:-2] + (9,))[..., 3:]
    return np.concatenate(
        (simulated_aruco_dist, simulated_optitrack_translation, rotation_feature),
        axis=-1,
    ).astype(np.float32)


def _interpolate_sequence(
    samples: np.ndarray,
    playback_speed: float,
    *,
    quaternion_slices: tuple[slice, ...] = (),
) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("Expected a two-dimensional sequence with at least two samples.")
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive.")

    num_intervals = len(samples) - 1
    output_intervals = max(round(num_intervals / playback_speed), 1)
    positions = np.linspace(0.0, num_intervals, output_intervals + 1)
    lower_indices = np.floor(positions).astype(np.int64)
    upper_indices = np.minimum(lower_indices + 1, len(samples) - 1)
    fraction = (positions - lower_indices)[:, None]
    interpolated = (
        (1.0 - fraction) * samples[lower_indices]
        + fraction * samples[upper_indices]
    )
    for quaternion_slice in quaternion_slices:
        first = samples[lower_indices, quaternion_slice]
        second = samples[upper_indices, quaternion_slice]
        second = np.where(
            np.sum(first * second, axis=-1, keepdims=True) < 0.0,
            -second,
            second,
        )
        quaternion = (1.0 - fraction) * first + fraction * second
        interpolated[:, quaternion_slice] = quaternion / np.linalg.norm(
            quaternion, axis=-1, keepdims=True
        )
    return interpolated.astype(samples.dtype, copy=False)


def _qpos_quaternion_slices(model: Any) -> tuple[slice, ...]:
    import mujoco

    quaternion_slices = []
    for joint_id, joint_type in enumerate(model.jnt_type):
        qpos_address = int(model.jnt_qposadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            quaternion_slices.append(slice(qpos_address + 3, qpos_address + 7))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            quaternion_slices.append(slice(qpos_address, qpos_address + 4))
    return tuple(quaternion_slices)


class SuccessfulTrajectoryWriter:
    """Append successful episodes to a simple, durable NPZ dataset."""

    def __init__(
        self,
        output_dir: Path,
        *,
        fps: float,
        task_name: str,
        include_images: bool = False,
        playback_speed: float = 1.0,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.fps = float(fps)
        self.include_images = bool(include_images)
        self.playback_speed = float(playback_speed)
        if self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive.")
        self.next_episode_index = self._find_next_episode_index()
        self._write_metadata()

    def _find_next_episode_index(self) -> int:
        indices = []
        for path in self.output_dir.glob("episode_*.npz"):
            try:
                indices.append(int(path.stem.removeprefix("episode_")))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def _write_metadata(self) -> None:
        metadata = {
            "format_version": 1,
            "task": self.task_name,
            "fps": self.fps,
            "playback_speed": self.playback_speed,
            "input_type": "absolute",
            "output_type": "absolute",
            "camera_features": self.include_images,
            "simulator_ground_truth": {
                "aruco_dist": (
                    "current gripper finger separation mapped to the real "
                    "ArUco-distance scale"
                ),
                "optitrack": "absolute end-effector pose",
            },
            "observation.state": {
                "shape": [len(STATE_FEATURE_NAMES)],
                "dtype": "float32",
                "names": list(STATE_FEATURE_NAMES),
            },
            "action": {
                "shape": [len(ACTION_FEATURE_NAMES)],
                "dtype": "float32",
                "names": list(ACTION_FEATURE_NAMES),
                "meaning": "next absolute state",
            },
        }
        if self.include_images:
            metadata[IMAGE_KEY] = {
                "shape": list(IMAGE_SHAPE),
                "dtype": "uint8",
                "names": ["height", "width", "channel"],
                "camera": SIM_CAMERA_NAME,
            }
        metadata_path = self.output_dir / "metadata.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.output_dir,
            prefix=".metadata_",
            suffix=".json",
            delete=False,
        ) as temporary_file:
            json.dump(metadata, temporary_file, indent=2)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, metadata_path)

    def save(
        self,
        states: list[np.ndarray],
        next_states: list[np.ndarray],
        images: np.ndarray | None = None,
    ) -> Path:
        if not states or len(states) != len(next_states):
            raise ValueError("A trajectory needs equally sized, non-empty state arrays.")
        if self.include_images:
            if images is None:
                raise ValueError("Image-enabled trajectories require camera images.")
            images = np.asarray(images, dtype=np.uint8)
            expected_shape = (len(states), *IMAGE_SHAPE)
            if images.shape != expected_shape:
                raise ValueError(
                    f"Expected {IMAGE_KEY} shape {expected_shape}, got {images.shape}."
                )
        elif images is not None:
            raise ValueError("Images were provided to a state-only trajectory writer.")

        episode_path = self.output_dir / f"episode_{self.next_episode_index:06d}.npz"
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=self.output_dir,
            prefix=".episode_",
            suffix=".npz",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            episode = {
                "observation.state": np.stack(states).astype(np.float32),
                "action": np.stack(next_states).astype(np.float32),
                "task": np.asarray(self.task_name),
                "fps": np.asarray(self.fps, dtype=np.float32),
            }
            if images is not None:
                episode[IMAGE_KEY] = images
            np.savez_compressed(temporary_file, **episode)
        os.replace(temporary_path, episode_path)
        self.next_episode_index += 1
        return episode_path


class SuccessfulTrajectoryCallback(BaseCallback):
    """Capture each vectorized rollout through its first successful step."""

    def __init__(
        self, output_dir: Path, *, task_name: str = DEFAULT_TASK_NAME
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.task_name = task_name
        self.writer: SuccessfulTrajectoryWriter | None = None
        self.current_observations: np.ndarray | None = None
        self.states: list[list[np.ndarray]] = []
        self.next_states: list[list[np.ndarray]] = []
        self.success_saved: list[bool] = []

    def _on_training_start(self) -> None:
        control_dt = float(self.training_env.get_attr("control_dt")[0])
        self.writer = SuccessfulTrajectoryWriter(
            self.output_dir,
            fps=1.0 / control_dt,
            task_name=self.task_name,
        )
        self.current_observations = np.asarray(self.model._last_obs).copy()
        num_envs = self.training_env.num_envs
        self.states = [[] for _ in range(num_envs)]
        self.next_states = [[] for _ in range(num_envs)]
        self.success_saved = [False] * num_envs
        print(f"Successful trajectory dataset: {self.writer.output_dir}")

    def _on_step(self) -> bool:
        if self.writer is None or self.current_observations is None:
            raise RuntimeError("Trajectory callback was not initialized.")

        new_observations = np.asarray(self.locals["new_obs"])
        dones = np.asarray(self.locals["dones"], dtype=bool)
        infos = self.locals["infos"]

        for env_index, (done, info) in enumerate(zip(dones, infos)):
            terminal_observation = info.get("terminal_observation") if done else None
            next_observation = (
                np.asarray(terminal_observation)
                if terminal_observation is not None
                else new_observations[env_index]
            )
            self.states[env_index].append(
                observation_to_absolute_state(self.current_observations[env_index])
            )
            self.next_states[env_index].append(
                observation_to_absolute_state(next_observation)
            )

            if info.get("success", False) and not self.success_saved[env_index]:
                episode_path = self.writer.save(
                    self.states[env_index], self.next_states[env_index]
                )
                self.success_saved[env_index] = True
                print(f"Saved successful trajectory to {episode_path}")

            if done:
                self.states[env_index] = []
                self.next_states[env_index] = []
                self.success_saved[env_index] = False

        self.current_observations = new_observations.copy()
        return True


def latest_checkpoint(path: Path) -> Path:
    """Returns a Brax checkpoint or the newest checkpoint inside a run."""
    return all_checkpoints(path)[-1]


def all_checkpoints(path: Path) -> list[Path]:
    """Returns every valid Brax checkpoint in ascending step order."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if (
        (path / "ppo_network_config.json").is_file()
        and (path / "commit_success.txt").is_file()
    ):
        return [path]

    checkpoints = [
        child
        for child in path.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and (child / "ppo_network_config.json").is_file()
        and (child / "commit_success.txt").is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No Brax PPO checkpoint found in {path}")
    return sorted(checkpoints, key=lambda checkpoint: int(checkpoint.name))


def load_mjx_policy(checkpoint_path: Path, env: Any):
    """Loads the deterministic Brax PPO policy used by the MJX trainer."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )
    params = ppo_checkpoint.load(checkpoint_path)
    return ppo_networks.make_inference_fn(network)(params, deterministic=True)


class MjxImageRenderer:
    """Render episodes with one persistent headless MuJoCo context."""

    def __init__(self, model: Any) -> None:
        import mujoco
        from new2 import CAMERA_JOINT_NAMES, CAMERA_ROT_JOINT_NAMES, CameraConfig

        self.model = model
        self.data = mujoco.MjData(model)
        camera_joint_names = CAMERA_JOINT_NAMES + CAMERA_ROT_JOINT_NAMES
        self.camera_qpos_indices = np.asarray(
            [
                model.jnt_qposadr[model.joint(joint_name).id]
                for joint_name in camera_joint_names
            ],
            dtype=np.int32,
        )
        camera_config = CameraConfig()
        self.camera_qpos = np.asarray(
            camera_config.pos + camera_config.euler,
            dtype=model.qpos0.dtype,
        )
        self.renderer = mujoco.Renderer(
            model,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )

    def _render_episode(self, qposes: np.ndarray) -> np.ndarray:
        import mujoco

        images = []
        for qpos in qposes:
            self.data.qpos[:] = qpos
            self.data.qpos[self.camera_qpos_indices] = self.camera_qpos
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=SIM_CAMERA_NAME)
            images.append(self.renderer.render().copy())
        return np.stack(images).astype(np.uint8, copy=False)

    def render(self, episodes: list[np.ndarray]) -> list[np.ndarray]:
        return [self._render_episode(qposes) for qposes in episodes]

    def close(self) -> None:
        self.renderer.close()


def _save_successful_rollouts(
    writer: SuccessfulTrajectoryWriter,
    observations: np.ndarray,
    next_observations: np.ndarray,
    successes: np.ndarray,
    *,
    episode_length: int,
    qposes: np.ndarray | None = None,
    next_qposes: np.ndarray | None = None,
    model: Any | None = None,
    image_renderer: MjxImageRenderer | None = None,
) -> list[Path]:
    if writer.include_images and (
        qposes is None or next_qposes is None or model is None
    ):
        raise ValueError(
            "Image-enabled trajectory saving requires current and next qposes "
            "plus a model."
        )

    successful_episodes = []
    post_success_steps = round(POST_SUCCESS_SECONDS * writer.fps)
    observation_quaternion_slices = (EE_QUAT_SLICE, WEED_QUAT_SLICE)
    qpos_quaternion_slices = (
        _qpos_quaternion_slices(model) if model is not None else ()
    )
    for episode in range(observations.shape[0]):
        success_steps = np.flatnonzero(
            np.asarray(successes[episode, :episode_length]) > 0.5
        )
        if not success_steps.size:
            continue
        length = min(
            int(success_steps[0]) + 1 + post_success_steps,
            observations.shape[1],
        )
        full_observations = np.concatenate(
            (
                observations[episode, :length],
                next_observations[episode, length - 1 : length],
            ),
            axis=0,
        )
        interpolated_observations = _interpolate_sequence(
            full_observations,
            writer.playback_speed,
            quaternion_slices=observation_quaternion_slices,
        )

        interpolated_qposes = None
        if qposes is not None and next_qposes is not None:
            full_qposes = np.concatenate(
                (
                    qposes[episode, :length],
                    next_qposes[episode, length - 1 : length],
                ),
                axis=0,
            )
            interpolated_qposes = _interpolate_sequence(
                full_qposes,
                writer.playback_speed,
                quaternion_slices=qpos_quaternion_slices,
            )[:-1]

        successful_episodes.append(
            (
                observation_to_absolute_state(interpolated_observations[:-1]),
                observation_to_absolute_state(interpolated_observations[1:]),
                interpolated_qposes,
            )
        )

    images = [None] * len(successful_episodes)
    if writer.include_images and successful_episodes:
        owns_renderer = image_renderer is None
        if image_renderer is None:
            image_renderer = MjxImageRenderer(model)
        try:
            images = image_renderer.render(
                [episode_qposes for _, _, episode_qposes in successful_episodes]
            )
        finally:
            if owns_renderer:
                image_renderer.close()

    saved_paths = []
    for (states, next_states, _), episode_images in zip(
        successful_episodes, images, strict=True
    ):
        saved_paths.append(
            writer.save(list(states), list(next_states), episode_images)
        )
    return saved_paths


class MjxTrainingTrajectoryCollector:
    """Appends successful PPO episodes plus a one-second release tail."""

    def __init__(
        self,
        env: Any,
        output_dir: Path,
        *,
        num_episodes: int,
        seed: int,
        task_name: str = DEFAULT_TASK_NAME,
        include_images: bool = False,
        playback_speed: float = 1.0,
    ) -> None:
        self.env = env
        self.num_episodes = int(num_episodes)
        self.seed = int(seed)
        self.writer = SuccessfulTrajectoryWriter(
            output_dir,
            fps=1.0 / float(env._config.ctrl_dt),
            task_name=task_name,
            include_images=include_images,
            playback_speed=playback_speed,
        )
        self.image_renderer = (
            MjxImageRenderer(env.mj_model) if include_images else None
        )
        self._rollout_fn = None

    def close(self) -> None:
        if self.image_renderer is not None:
            self.image_renderer.close()
            self.image_renderer = None

    def _build_rollout_fn(self, make_policy):
        import jax
        import jax.numpy as jnp

        def rollout(reset_rng, policy_rng, params):
            policy = make_policy(params, deterministic=False)
            state = self.env.reset(reset_rng)

            def step(carry, _):
                current_state, rng = carry
                rng, action_rng = jax.random.split(rng)
                action, _ = policy(current_state.obs, action_rng)
                action = jnp.where(
                    current_state.done > 0.0,
                    jnp.zeros_like(action),
                    action,
                )
                next_state = self.env.step(current_state, action)
                transition = (
                    current_state.obs,
                    next_state.obs,
                    next_state.metrics["success"],
                    current_state.data.qpos,
                    next_state.data.qpos,
                )
                return (next_state, rng), transition

            _, transitions = jax.lax.scan(
                step,
                (state, policy_rng),
                None,
                length=(
                    self.env._config.episode_length
                    + round(POST_SUCCESS_SECONDS * self.writer.fps)
                ),
            )
            return transitions

        return jax.jit(jax.vmap(rollout, in_axes=(0, 0, None)))

    def __call__(self, step: int, make_policy, params) -> list[Path]:
        if step == 0:
            return []

        import jax

        if self._rollout_fn is None:
            self._rollout_fn = self._build_rollout_fn(make_policy)
        step_rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), step)
        reset_rng, policy_rng = jax.random.split(step_rng)
        reset_rngs = jax.random.split(reset_rng, self.num_episodes)
        policy_rngs = jax.random.split(policy_rng, self.num_episodes)
        observations, next_observations, successes, qposes, next_qposes = jax.device_get(
            self._rollout_fn(reset_rngs, policy_rngs, params)
        )
        saved_paths = _save_successful_rollouts(
            self.writer,
            observations,
            next_observations,
            successes,
            episode_length=self.env._config.episode_length,
            qposes=qposes,
            next_qposes=next_qposes,
            model=self.env.mj_model,
            image_renderer=self.image_renderer,
        )
        print(
            f"step={step} saved_successful_trajectories="
            f"{len(saved_paths)}/{self.num_episodes}"
        )
        return saved_paths


def collect_mjx_checkpoint_trajectories(
    checkpoint_paths: list[Path],
    output_dir: Path,
    *,
    num_episodes: int,
    seed: int,
    task_name: str,
    include_images: bool = False,
    playback_speed: float = 1.0,
) -> list[Path]:
    """Appends successful training-style rollouts from multiple checkpoints."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    from train_weed_mjx_warp import WeedPickMjxWarpEnv, default_config

    env = WeedPickMjxWarpEnv(
        default_config(
            num_envs=num_episodes,
            force_release_after_hold=True,
        )
    )
    collector = MjxTrainingTrajectoryCollector(
        env,
        output_dir,
        num_episodes=num_episodes,
        seed=seed,
        task_name=task_name,
        include_images=include_images,
        playback_speed=playback_speed,
    )
    network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )
    make_policy = ppo_networks.make_inference_fn(network)

    saved_paths = []
    try:
        for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            print(
                f"checkpoint={checkpoint_index}/{len(checkpoint_paths)} "
                f"path={checkpoint_path}"
            )
            params = ppo_checkpoint.load(checkpoint_path)
            saved_paths.extend(
                collector(int(checkpoint_path.name), make_policy, params)
            )
    finally:
        collector.close()
    return saved_paths


def collect_mjx_trajectories(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num_episodes: int,
    seed: int,
    task_name: str,
    include_images: bool = False,
    playback_speed: float = 1.0,
) -> list[Path]:
    """Saves successful MJX episodes plus a one-second release tail."""
    import jax
    import jax.numpy as jnp

    from train_weed_mjx_warp import WeedPickMjxWarpEnv, default_config

    env_config = default_config(
        num_envs=num_episodes,
        force_release_after_hold=True,
    )
    env = WeedPickMjxWarpEnv(env_config)
    policy = load_mjx_policy(checkpoint_path, env)

    def rollout(reset_rng, policy_rng):
        state = env.reset(reset_rng)

        def step(carry, _):
            current_state, rng = carry
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(current_state.obs, action_rng)
            action = jnp.where(
                current_state.done > 0.0,
                jnp.zeros_like(action),
                action,
            )
            next_state = env.step(current_state, action)
            transition = (
                current_state.obs,
                next_state.obs,
                next_state.metrics["success"],
                current_state.data.qpos,
                next_state.data.qpos,
            )
            return (next_state, rng), transition

        _, transitions = jax.lax.scan(
            step,
            (state, policy_rng),
            None,
            length=(
                env_config.episode_length
                + round(POST_SUCCESS_SECONDS / env_config.ctrl_dt)
            ),
        )
        return transitions

    reset_rngs = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    policy_rngs = jax.random.split(jax.random.PRNGKey(seed + 1), num_episodes)
    observations, next_observations, successes, qposes, next_qposes = jax.device_get(
        jax.jit(jax.vmap(rollout))(reset_rngs, policy_rngs)
    )

    writer = SuccessfulTrajectoryWriter(
        output_dir,
        fps=1.0 / float(env_config.ctrl_dt),
        task_name=task_name,
        include_images=include_images,
        playback_speed=playback_speed,
    )
    return _save_successful_rollouts(
        writer,
        observations,
        next_observations,
        successes,
        episode_length=env_config.episode_length,
        qposes=qposes,
        next_qposes=next_qposes,
        model=env.mj_model,
    )


def parse_args() -> argparse.Namespace:
    from train_weed_mjx_warp import DEFAULT_RUN_NAME, DEFAULT_RUN_ROOT

    parser = argparse.ArgumentParser(
        description="Save successful trajectories from an MJX-Warp PPO checkpoint."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Collect from every checkpoint in the selected run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Trajectory speed multiplier; use 0.3 for 70%% slower motion.",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help=(
            f"Render {IMAGE_KEY} from {SIM_CAMERA_NAME} at "
            f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}."
        ),
    )
    args = parser.parse_args()
    args.run_root = DEFAULT_RUN_ROOT
    return args


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")

    checkpoint_root = args.checkpoint or args.run_root / args.run_name
    checkpoint_paths = all_checkpoints(checkpoint_root)
    checkpoint_path = checkpoint_paths[-1]
    run_dir = (
        checkpoint_path.parent if checkpoint_path.name.isdigit() else checkpoint_path
    )
    output_dir = args.output or run_dir / "successful_trajectories"
    if args.all_checkpoints:
        saved_paths = collect_mjx_checkpoint_trajectories(
            checkpoint_paths,
            output_dir,
            num_episodes=args.episodes,
            seed=args.seed,
            task_name=args.task_name,
            include_images=args.images,
            playback_speed=args.playback_speed,
        )
        candidate_episodes = len(checkpoint_paths) * args.episodes
        print(f"Loaded checkpoints: {len(checkpoint_paths)}")
    else:
        saved_paths = collect_mjx_trajectories(
            checkpoint_path,
            output_dir,
            num_episodes=args.episodes,
            seed=args.seed,
            task_name=args.task_name,
            include_images=args.images,
            playback_speed=args.playback_speed,
        )
        candidate_episodes = args.episodes
        print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Successful episodes: {len(saved_paths)}/{candidate_episodes}")
    print(f"Dataset: {output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
