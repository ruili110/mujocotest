# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lerobot==0.3.3",
#     "numpy>=2.2.6,<3",
# ]
# ///
"""Convert successful NPZ trajectories to a LeRobot dataset."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_dataset import (
    ACTION_FEATURE_NAMES,
    DEFAULT_TASK_NAME,
    IMAGE_KEY,
    IMAGE_SHAPE,
    STATE_FEATURE_NAMES,
    gripper_width_to_aruco_distance,
)


STATE_KEY = "observation.state"
ACTION_KEY = "action"
TARGET_FPS = 25
EXPECTED_CODEBASE_VERSION = "v2.1"


def _load_lerobot_api() -> tuple[type, dict[str, Any], str]:
    try:
        from lerobot.common.datasets.lerobot_dataset import (
            CODEBASE_VERSION,
            LeRobotDataset,
        )
        from lerobot.common.datasets.utils import DEFAULT_FEATURES

        return LeRobotDataset, DEFAULT_FEATURES, str(CODEBASE_VERSION)
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

        return LeRobotDataset, {}, str(CODEBASE_VERSION)


def _remove_unused_default_features(features: dict) -> None:
    for feature_name in ("reward", "done", "is_terminal"):
        features.pop(feature_name, None)


def _load_transitions(
    episode_path: Path, target_fps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(episode_path, allow_pickle=False) as episode:
        states = np.asarray(episode[STATE_KEY], dtype=np.float32)
        actions = np.asarray(episode[ACTION_KEY], dtype=np.float32)
        images = (
            np.asarray(episode[IMAGE_KEY], dtype=np.uint8)
            if IMAGE_KEY in episode
            else None
        )
        source_fps = float(episode["fps"])

    expected_shape = (len(STATE_FEATURE_NAMES),)
    if states.ndim != 2 or states.shape[1:] != expected_shape:
        raise ValueError(
            f"{episode_path}: expected {STATE_KEY} shape (N, {expected_shape[0]}), "
            f"got {states.shape}."
        )
    if actions.shape != states.shape:
        raise ValueError(
            f"{episode_path}: {ACTION_KEY} shape {actions.shape} does not match "
            f"{STATE_KEY} shape {states.shape}."
        )
    if len(states) == 0:
        raise ValueError(f"{episode_path}: episode is empty.")
    if images is not None:
        expected_image_shape = (len(states), *IMAGE_SHAPE)
        if images.shape != expected_image_shape:
            raise ValueError(
                f"{episode_path}: expected {IMAGE_KEY} shape "
                f"{expected_image_shape}, got {images.shape}."
            )

    gripper_values = np.concatenate((states[:, 0], actions[:, 0]))
    if np.max(np.abs(gripper_values)) < 1.0:
        states = states.copy()
        actions = actions.copy()
        states[:, 0] = gripper_width_to_aruco_distance(states[:, 0])
        actions[:, 0] = gripper_width_to_aruco_distance(actions[:, 0])

    fps_ratio = source_fps / target_fps
    stride = round(fps_ratio)
    if stride < 1 or not np.isclose(fps_ratio, stride):
        raise ValueError(
            f"{episode_path}: source FPS {source_fps:g} is not an integer "
            f"multiple of target FPS {target_fps}."
        )
    if stride == 1:
        return states, actions, images

    if len(states) > 1 and not np.allclose(actions[:-1], states[1:]):
        raise ValueError(
            f"{episode_path}: actions are not next states, so safe downsampling "
            "is not possible."
        )

    full_state_sequence = np.concatenate((states, actions[-1:]), axis=0)
    sampled_indices = list(range(0, len(full_state_sequence), stride))
    if sampled_indices[-1] != len(full_state_sequence) - 1:
        sampled_indices.append(len(full_state_sequence) - 1)
    sampled_states = full_state_sequence[sampled_indices]
    sampled_images = images[sampled_indices[:-1]] if images is not None else None
    return sampled_states[:-1], sampled_states[1:], sampled_images


def _add_frame(
    dataset,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    image: np.ndarray | None = None,
) -> None:
    frame = {
        STATE_KEY: state.astype(np.float32, copy=False),
        ACTION_KEY: action.astype(np.float32, copy=False),
    }
    if image is not None:
        frame[IMAGE_KEY] = image
    if "task" in inspect.signature(dataset.add_frame).parameters:
        dataset.add_frame(frame, task=task)
    else:
        frame["task"] = task
        dataset.add_frame(frame)


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    task: str,
    fps: int,
) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    episode_paths = sorted(input_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {input_dir}.")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new path."
        )

    lerobot_dataset, default_features, codebase_version = _load_lerobot_api()
    if codebase_version != EXPECTED_CODEBASE_VERSION:
        raise RuntimeError(
            f"Expected LeRobot dataset format {EXPECTED_CODEBASE_VERSION}, "
            f"but the installed package writes {codebase_version}. Run this "
            "file with `uv run --script convert_npz_to_lerobot.py ...` to use "
            "the pinned compatible version."
        )
    with np.load(episode_paths[0], allow_pickle=False) as first_episode:
        include_images = IMAGE_KEY in first_episode

    features = {
        **default_features,
        STATE_KEY: {
            "shape": (len(STATE_FEATURE_NAMES),),
            "dtype": "float32",
            "names": list(STATE_FEATURE_NAMES),
        },
        ACTION_KEY: {
            "shape": (len(ACTION_FEATURE_NAMES),),
            "dtype": "float32",
            "names": list(ACTION_FEATURE_NAMES),
        },
    }
    if include_images:
        features[IMAGE_KEY] = {
            "shape": IMAGE_SHAPE,
            "dtype": "image",
            "names": ["height", "width", "channel"],
        }
    _remove_unused_default_features(features)
    dataset = lerobot_dataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        fps=fps,
        features=features,
        use_videos=False,
    )

    total_frames = 0
    for episode_path in episode_paths:
        states, actions, images = _load_transitions(episode_path, fps)
        if (images is not None) != include_images:
            raise ValueError(
                f"{episode_path}: image presence does not match the first episode."
            )
        image_frames = images if images is not None else [None] * len(states)
        for state, action, image in zip(
            states, actions, image_frames, strict=True
        ):
            _add_frame(dataset, state, action, task, image)
        dataset.save_episode()
        total_frames += len(states)
        print(f"Saved {episode_path.name}: {len(states)} frames")

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    print(f"Converted {len(episode_paths)} episodes and {total_frames} frames.")
    print(f"LeRobot format: {codebase_version}")
    print(f"LeRobot dataset: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert successful NPZ trajectories to LeRobot."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK_NAME)
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.input,
        args.output,
        task=args.task,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
