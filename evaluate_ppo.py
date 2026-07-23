from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from weed_pick_env import WeedPickEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained weed-picking PPO model.")
    parser.add_argument("model", type=Path, help="Path to an SB3 PPO .zip model.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--stochastic", action="store_true")
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

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    successes = 0

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            episode_length = 0

            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1

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
        env.close()

    print(f"mean_return={np.mean(episode_returns):.3f}")
    print(f"mean_length={np.mean(episode_lengths):.1f}")
    print(f"success_rate={successes / args.episodes:.1%}")


if __name__ == "__main__":
    main()
