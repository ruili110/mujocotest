# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dm-control>=1.0.43",
#     "imageio-ffmpeg>=0.6.0",
#     "jax[cuda12]==0.8.2",
#     "mediapy>=1.2.4",
#     "mujoco==3.11.0",
#     "mujoco-mjx[warp]==3.11.0",
#     "playground==0.2.0",
#     "wandb>=0.22.0",
# ]
# ///
"""Render a deterministic video from a weed-picking Brax PPO checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import imageio_ffmpeg
import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import wandb

from train_weed_mjx_warp import (
    DEFAULT_RUN_NAME,
    DEFAULT_RUN_ROOT,
    WeedPickMjxWarpEnv,
    default_config,
)


DEFAULT_EPISODES = 10
DEFAULT_VIDEO_ROOT = Path(__file__).resolve().parent / "runs"
RESET_PAUSE_SECONDS = 0.5
POST_SUCCESS_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trained weed-picking policy."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upload-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="weed-picking")
    return parser.parse_args()


def latest_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if (
        (path / "ppo_network_config.json").is_file()
        and (path / "commit_success.txt").is_file()
    ):
        return path

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
    return max(checkpoints, key=lambda checkpoint: int(checkpoint.name))


def rollout_episodes(
    env: WeedPickMjxWarpEnv,
    policy,
    seed: int,
    episode_length: int,
    num_episodes: int,
) -> tuple[list[list], list[dict[str, float | bool]]]:
    def rollout(reset_rng, policy_rng):
        state = env.reset(reset_rng)
        empty_data = state.data.__class__(
            **{name: None for name in state.data.__annotations__}
        )
        empty_state = state.__class__(
            **{name: None for name in state.__annotations__}
        ).replace(data=empty_data)

        def step(carry, _):
            current_state, rng = carry
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(current_state.obs, action_rng)
            action = jp.where(
                current_state.done > 0.0,
                jp.zeros_like(action),
                action,
            )
            next_state = env.step(current_state, action)
            render_state = empty_state.tree_replace(
                {
                    "data.qpos": next_state.data.qpos,
                    "data.qvel": next_state.data.qvel,
                    "data.time": next_state.data.time,
                    "data.ctrl": next_state.data.ctrl,
                    "data.mocap_pos": next_state.data.mocap_pos,
                    "data.mocap_quat": next_state.data.mocap_quat,
                    "data.xfrc_applied": next_state.data.xfrc_applied,
                }
            )
            episode_data = {
                "reward": next_state.reward,
                "done": next_state.done,
                "success": next_state.metrics["success"],
                "grasp_once": next_state.metrics["grasp_once"],
                "target_reached_once": next_state.metrics["target_reached_once"],
                "target_hold_count": next_state.info["target_hold_count"],
            }
            return (next_state, rng), (render_state, episode_data)

        _, result = jax.lax.scan(
            step,
            (state, policy_rng),
            None,
            length=episode_length,
        )
        return result

    reset_rngs = jp.stack(
        [jax.random.PRNGKey(seed + episode) for episode in range(num_episodes)]
    )
    policy_rngs = jp.stack(
        [
            jax.random.PRNGKey(seed + num_episodes + episode)
            for episode in range(num_episodes)
        ]
    )
    trajectories, episode_data = jax.device_get(
        jax.jit(jax.vmap(rollout))(reset_rngs, policy_rngs)
    )

    episode_trajectories = []
    episode_results = []
    for episode in range(num_episodes):
        done_steps = np.flatnonzero(episode_data["done"][episode])
        length = int(done_steps[0] + 1) if done_steps.size else episode_length
        succeeded = bool(np.any(episode_data["success"][episode, :length]))
        render_length = length
        if succeeded:
            post_success_steps = round(POST_SUCCESS_SECONDS / env.dt)
            render_length = min(length + post_success_steps, episode_length)
        episode_trajectories.append(
            [
                jax.tree.map(
                    lambda value, episode=episode, step=step: value[episode, step],
                    trajectories,
                )
                for step in range(render_length)
            ]
        )
        episode_results.append(
            {
                "return": float(np.sum(episode_data["reward"][episode, :length])),
                "length": float(length),
                "success": succeeded,
                "grasped": bool(
                    np.any(episode_data["grasp_once"][episode, :length])
                ),
                "target_reached": bool(
                    np.any(episode_data["target_reached_once"][episode, :length])
                ),
                "max_target_hold": float(
                    np.max(episode_data["target_hold_count"][episode, :length])
                ),
            }
        )

    return episode_trajectories, episode_results


def load_policy(checkpoint_path: Path, env: WeedPickMjxWarpEnv):
    network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )
    make_policy = ppo_networks.make_inference_fn(network)
    params = ppo_checkpoint.load(checkpoint_path)
    return make_policy(params, deterministic=True)


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    checkpoint_root = args.checkpoint or DEFAULT_RUN_ROOT / args.run_name
    checkpoint_path = latest_checkpoint(checkpoint_root)
    output_path = args.output or DEFAULT_VIDEO_ROOT / args.run_name / "weed_policy.mp4"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_config = default_config(
        num_envs=args.episodes,
        force_release_after_hold=True,
    )
    env = WeedPickMjxWarpEnv(env_config)
    policy = jax.jit(load_policy(checkpoint_path, env))
    trajectories, episode_results = rollout_episodes(
        env,
        policy,
        args.seed,
        env_config.episode_length,
        args.episodes,
    )

    render_every = 2
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.mj_model, camera)
    camera.lookat[:] = [0.28, 0.0, 0.34]
    camera.distance = 1.4
    camera.azimuth = 180
    camera.elevation = -30
    fps = 1.0 / env.dt / render_every
    reset_pause_frames = max(round(RESET_PAUSE_SECONDS * fps), 1)
    frames = []
    for trajectory in trajectories:
        episode_frames = env.render(
            trajectory[::render_every],
            height=480,
            width=640,
            camera=camera,
            scene_option=scene_option,
        )
        frames.extend([episode_frames[0]] * reset_pause_frames)
        frames.extend(episode_frames)

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    media.write_video(str(output_path), frames, fps=fps)

    for episode, result in enumerate(episode_results, start=1):
        print(
            f"episode={episode} return={result['return']:.3f} "
            f"length={result['length']:.0f} success={result['success']} "
            f"grasped={result['grasped']} "
            f"target_reached={result['target_reached']} "
            f"max_target_hold={result['max_target_hold']:.0f}"
        )
    mean_return = float(np.mean([result["return"] for result in episode_results]))
    mean_length = float(np.mean([result["length"] for result in episode_results]))
    success_rate = float(
        np.mean([result["success"] for result in episode_results])
    )
    print(f"mean_return={mean_return:.3f}")
    print(f"mean_length={mean_length:.1f}")
    print(f"success_rate={success_rate:.1%}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved video: {output_path}")

    if args.upload_wandb:
        with wandb.init(
            project=args.wandb_project,
            job_type="evaluation",
            name=f"checkpoint-{checkpoint_path.name}-video",
        ) as run:
            run.log(
                {
                    "evaluation/mean_return": mean_return,
                    "evaluation/mean_length": mean_length,
                    "evaluation/success_rate": success_rate,
                    "policy_video": wandb.Video(str(output_path), format="mp4"),
                }
            )
        print(f"Uploaded video to W&B project: {args.wandb_project}")


if __name__ == "__main__":
    main()
