from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from trajectory_dataset import SuccessfulTrajectoryCallback
from weed_pick_env import WeedPickEnv, run_env_checks


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ppo_weed_pick"


class TaskMetricsCallback(BaseCallback):
    """Record rolling task outcomes through the standard SB3 logger."""

    def __init__(self, window_size: int = 100) -> None:
        super().__init__()
        self.successes = deque(maxlen=window_size)
        self.grasps = deque(maxlen=window_size)
        self.clearances = deque(maxlen=window_size)
        self.left_contacts = deque(maxlen=window_size)
        self.right_contacts = deque(maxlen=window_size)
        self.reach_distances = deque(maxlen=window_size)
        self.lift_amounts = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            self.successes.append(float(info.get("success", False)))
            self.grasps.append(float(info.get("grasp_once", False)))
            self.clearances.append(
                float(info.get("weed_clears_pot_once", False))
            )
            self.left_contacts.append(float(info.get("left_contact_once", False)))
            self.right_contacts.append(float(info.get("right_contact_once", False)))
            self.reach_distances.append(float(info.get("reach_distance", 0.0)))
            self.lift_amounts.append(float(info.get("lift_amount", 0.0)))

        if self.successes:
            self.logger.record("task/success_rate", sum(self.successes) / len(self.successes))
            self.logger.record(
                "task/grasp_once_rate",
                sum(self.grasps) / len(self.grasps),
            )
            self.logger.record(
                "task/weed_clears_pot_once_rate",
                sum(self.clearances) / len(self.clearances),
            )
            self.logger.record(
                "task/left_contact_once_rate",
                sum(self.left_contacts) / len(self.left_contacts),
            )
            self.logger.record(
                "task/right_contact_once_rate",
                sum(self.right_contacts) / len(self.right_contacts),
            )
            self.logger.record(
                "task/final_reach_distance",
                sum(self.reach_distances) / len(self.reach_distances),
            )
            self.logger.record(
                "task/final_lift_amount",
                sum(self.lift_amounts) / len(self.lift_amounts),
            )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the Franka weed-picking task.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="SB3 device: cpu, cuda, or auto.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--wandb-project", default=None, help="Enable W&B with this project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B team or username.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name.")
    parser.add_argument(
        "--successful-trajectories-dir",
        type=Path,
        default=None,
        help=(
            "Camera-free successful trajectory dataset directory. Defaults to "
            "<output-dir>/successful_trajectories."
        ),
    )
    parser.add_argument(
        "--no-save-successful-trajectories",
        action="store_true",
        help="Disable successful trajectory collection.",
    )
    parser.add_argument(
        "--wandb-save-code",
        action="store_true",
        help="Upload source code; disabled by default because this repo contains large meshes.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
        help="Use offline mode when the training machine cannot reach W&B.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size < 2 or rollout_size % args.batch_size != 0:
        raise ValueError(
            f"--batch-size must divide n_envs * n_steps ({rollout_size}); "
            f"got {args.batch_size}."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    evaluation_dir = output_dir / "evaluations"
    monitor_dir = output_dir / "monitor"
    tensorboard_dir = output_dir / "tensorboard"

    for directory in (
        checkpoint_dir,
        best_model_dir,
        evaluation_dir,
        monitor_dir,
        tensorboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if args.check_env:
        run_env_checks()

    wandb_run = None
    if args.wandb_project is not None:
        import wandb

        wandb_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=wandb_config,
            sync_tensorboard=True,
            save_code=args.wandb_save_code,
            dir=str(output_dir),
        )
        tensorboard_dir = Path(wandb_run.dir) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        WeedPickEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(monitor_dir),
        monitor_kwargs={
            "info_keywords": (
                "success",
                "is_grasped",
                "grasp_once",
                "left_contact_once",
                "right_contact_once",
                "reach_distance",
                "target_distance",
                "target_reached",
                "robot_speed",
                "is_robot_static",
                "lift_amount",
                "weed_clears_pot",
                "weed_clears_pot_once",
                "target_reached_once",
                "left_grasp_force",
                "right_grasp_force",
                "left_grasp_force_angle",
                "right_grasp_force_angle",
                "success_hold_count",
            ),
        },
    )
    eval_env = None

    callbacks = [TaskMetricsCallback()]
    if not args.no_save_successful_trajectories:
        trajectory_dir = (
            args.successful_trajectories_dir.expanduser().resolve()
            if args.successful_trajectories_dir is not None
            else (output_dir / "successful_trajectories").resolve()
        )
        callbacks.append(SuccessfulTrajectoryCallback(trajectory_dir))

    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback

        wandb_callback_kwargs = {"verbose": 1}
        if args.checkpoint_freq > 0:
            wandb_callback_kwargs.update(
                model_save_path=str(output_dir / "wandb_models"),
                model_save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            )
        callbacks.append(WandbCallback(**wandb_callback_kwargs))
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="ppo_weed_pick",
            )
        )

    if args.eval_freq > 0:
        eval_env = make_vec_env(WeedPickEnv, n_envs=1, seed=args.seed + 10_000)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(evaluation_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    try:
        if args.resume is not None:
            model_path = args.resume.expanduser().resolve()
            model = PPO.load(
                model_path,
                env=train_env,
                device=args.device,
                tensorboard_log=str(tensorboard_dir),
            )
            print(f"Resuming PPO from {model_path}")
        else:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                target_kl=args.target_kl,
                ent_coef=0.0,
                policy_kwargs={"net_arch": [256, 256]},
                seed=args.seed,
                device=args.device,
                tensorboard_log=str(tensorboard_dir),
                verbose=1,
            )

        print(f"Training outputs: {output_dir}")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=args.resume is None,
            tb_log_name=wandb_run.id if wandb_run is not None else "PPO",
        )
        final_path = output_dir / "ppo_weed_pick_final"
        model.save(final_path)
        print(f"Saved final PPO model to {final_path}.zip")
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
