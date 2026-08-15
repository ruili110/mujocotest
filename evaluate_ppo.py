from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO

from weed_pick_env import WeedPickEnv


RESET_PAUSE_SECONDS = 0.5


class VideoRecorder:
    """Record an external-camera view of a WeedPickEnv rollout."""

    def __init__(self, env: WeedPickEnv, output_path: Path) -> None:
        import cv2

        self.cv2 = cv2
        self.output_path = output_path.expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        width = 640
        height = 480
        self.fps = max(round(1.0 / env.control_dt), 1)
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self.camera)
        self.camera.lookat[:] = [0.28, 0.0, 0.34]
        self.camera.distance = 1.4
        self.camera.azimuth = 180
        self.camera.elevation = -30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height),
        )
        if not self.writer.isOpened():
            self.renderer.close()
            raise RuntimeError(f"Could not open video output {self.output_path}")

    def capture(self, env: WeedPickEnv, frame_count: int = 1) -> None:
        self.renderer.update_scene(env.data, camera=self.camera)
        rgb = self.renderer.render()
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        for _ in range(frame_count):
            self.writer.write(bgr)

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained weed-picking PPO model.")
    parser.add_argument("model", type=Path, help="Path to an SB3 PPO .zip model.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional MP4 output path for recording the evaluated episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO model not found: {model_path}")

    env = WeedPickEnv()
    model = PPO.load(model_path, device="cpu")
    recorder = VideoRecorder(env, args.video) if args.video is not None else None

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    successes = 0

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            if recorder is not None:
                reset_frame_count = max(
                    round(RESET_PAUSE_SECONDS * recorder.fps),
                    1,
                )
                recorder.capture(env, frame_count=reset_frame_count)
            episode_return = 0.0
            episode_length = 0

            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if recorder is not None:
                    recorder.capture(env)

                if terminated or truncated:
                    successes += int(info["success"])
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    print(
                        f"episode={episode + 1} return={episode_return:.3f} "
                        f"length={episode_length} success={info['success']}"
                    )
                    break
    finally:
        if recorder is not None:
            recorder.close()
        env.close()

    print(f"mean_return={np.mean(episode_returns):.3f}")
    print(f"mean_length={np.mean(episode_lengths):.1f}")
    print(f"success_rate={successes / args.episodes:.1%}")
    if recorder is not None:
        print(f"video={recorder.output_path}")


if __name__ == "__main__":
    main()
