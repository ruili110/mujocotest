from __future__ import annotations

import argparse
import queue
import threading
import time

import numpy as np

from new2 import FrankaUmiCameraEnv


HOME_ARM = np.array([0.0, -0.405, 0.0, -2.25, 0.0, 1.28, 1.0], dtype=float)
DEFAULT_GRIPPER = 0.03
DEFAULT_CAMERA_POS = np.array([0.0, -0.12249176, 0.05800533], dtype=float)
DEFAULT_CAMERA_EULER = np.array([-0.6751, 0.0, 0.0], dtype=float)


def print_help() -> None:
    print(
        """
Manual RL placeholder controls

Commands:
  h                         show this help
  p                         print current qpos/qvel/ctrl/camera pose
  home                      send robot to HOME_ARM
  arm j1 j2 j3 j4 j5 j6 j7  set 7 Panda joint targets
  delta i value             add value to arm joint i, where i is 1..7
  grip value                set gripper opening, usually 0.0..0.05
  cam x y z                 set camera xyz
  rot roll pitch yaw        set camera roll/pitch/yaw in radians
  step n                    step simulation n times with current controls
  img                       render one camera image and print its shape
  q                         quit

Examples:
  arm 0 -0.405 0 -2.25 0 1.28 1
  delta 2 0.05
  grip 0.03
  cam 0 -0.12249176 0.05800533
  rot -0.6751 0 0
"""
    )


def print_state(obs: dict[str, np.ndarray], env: FrankaUmiCameraEnv) -> None:
    print("actuator_names =", env.actuator_names)
    print("qpos =", np.round(obs["qpos"], 5))
    print("qvel =", np.round(obs["qvel"], 5))
    print("ctrl =", np.round(obs["ctrl"], 5))
    print("camera_pos =", np.round(obs["camera_pos"], 5))
    print("camera_euler =", np.round(obs["camera_euler"], 5))


def parse_floats(parts: list[str], expected: int) -> list[float]:
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} numbers, got {len(parts)}.")
    return [float(value) for value in parts]


# =============================================================================
# RL_AGENT_TODO: future RL agent integration starts here.
#
# When you have a trained model, usually you only need to edit these functions:
#   1. load_rl_agent()
#   2. build_agent_observation()
#   3. agent_output_to_env_action()
#
# The refresh loop is:
#   obs = env.observe(...)
#   agent_obs = build_agent_observation(obs)
#   agent_output = agent.predict(agent_obs)
#   action = agent_output_to_env_action(agent_output)
#   obs = env.step(action, ...)
#   viewer.sync()  # only when --viewer is enabled
# =============================================================================


def load_rl_agent() -> object | None:
    """Load your trained RL model here.

    Example for stable-baselines3:
        from stable_baselines3 import PPO
        return PPO.load("your_model.zip")

    Keep returning None while you are using the manual placeholder.
    """
    return None


def build_agent_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Convert new2.py observation dict into your model input.

    Change this to exactly match the observation format used during training.
    For state-only RL, qpos+qvel is a common starting point.
    For visual policies, you may use obs["image"] and normalize/resize it here.
    """
    return np.concatenate([obs["qpos"], obs["qvel"]]).astype(np.float32)


def predict_with_agent(agent: object | None, agent_obs: np.ndarray) -> np.ndarray:
    """Run model inference.

    Replace the placeholder with your model's predict call.
    For stable-baselines3:
        action, _ = agent.predict(agent_obs, deterministic=True)
        return np.asarray(action, dtype=float)
    """
    if agent is None:
        # Placeholder action: hold the home pose and keep the gripper slightly open.
        return np.concatenate([HOME_ARM, [DEFAULT_GRIPPER]])

    action, _ = agent.predict(agent_obs, deterministic=True)  # type: ignore[attr-defined]
    return np.asarray(action, dtype=float)


def agent_output_to_env_action(agent_output: np.ndarray) -> dict[str, object]:
    """Convert model output into the action format accepted by FrankaUmiCameraEnv.

    Current expected placeholder format:
        [joint1..joint7, gripper]

    If your trained model outputs a different action, change only this adapter.
    """
    if agent_output.shape[0] < 8:
        raise ValueError("Expected agent output with at least 8 values: 7 arm joints + gripper.")

    return {
        "arm": agent_output[:7],
        "gripper": float(agent_output[7]),
        "camera_pos": DEFAULT_CAMERA_POS,
        "camera_euler": DEFAULT_CAMERA_EULER,
    }


def run_agent_step(
    env: FrankaUmiCameraEnv,
    agent: object | None,
    obs: dict[str, np.ndarray],
    n_substeps: int,
    render: bool,
) -> dict[str, np.ndarray]:
    """One RL refresh: observe -> predict -> control -> step."""
    agent_obs = build_agent_observation(obs)
    agent_output = predict_with_agent(agent, agent_obs)
    action = agent_output_to_env_action(agent_output)
    return env.step(action, n_substeps=n_substeps, render=render)


class ManualController:
    def __init__(self, env: FrankaUmiCameraEnv) -> None:
        self.env = env
        obs = env.reset(render=False)
        self.current_arm = HOME_ARM.copy()
        self.current_gripper = DEFAULT_GRIPPER
        self.current_camera_pos = DEFAULT_CAMERA_POS.copy()
        self.current_camera_euler = DEFAULT_CAMERA_EULER.copy()

        env.set_control(
            arm=self.current_arm,
            gripper=self.current_gripper,
            camera_pos=self.current_camera_pos,
            camera_euler=self.current_camera_euler,
        )

    def handle(self, raw: str) -> bool:
        """Handle one command. Returns False when the program should quit."""
        if not raw:
            return True

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"q", "quit", "exit"}:
            return False

        if cmd in {"h", "help"}:
            print_help()
            return True

        if cmd == "p":
            obs = self.env.observe(render=False)
            print_state(obs, self.env)
            return True

        if cmd == "home":
            self.current_arm = HOME_ARM.copy()
            self.env.set_control(arm=self.current_arm, gripper=self.current_gripper)

        elif cmd == "arm":
            self.current_arm = np.array(parse_floats(args, 7), dtype=float)
            self.env.set_control(arm=self.current_arm)

        elif cmd == "delta":
            if len(args) != 2:
                raise ValueError("Usage: delta i value")
            joint_index = int(args[0]) - 1
            if not 0 <= joint_index < 7:
                raise ValueError("Joint index must be 1..7.")
            self.current_arm[joint_index] += float(args[1])
            self.env.set_control(arm=self.current_arm)

        elif cmd in {"grip", "gripper"}:
            self.current_gripper = float(args[0])
            self.env.set_control(gripper=self.current_gripper)

        elif cmd in {"cam", "camera"}:
            self.current_camera_pos = np.array(parse_floats(args, 3), dtype=float)
            self.env.set_control(camera_pos=self.current_camera_pos)

        elif cmd in {"rot", "euler"}:
            self.current_camera_euler = np.array(parse_floats(args, 3), dtype=float)
            self.env.set_control(camera_euler=self.current_camera_euler)

        elif cmd == "step":
            n_steps = int(args[0]) if args else 1
            self.env.step(None, n_substeps=n_steps, render=False)
            print("stepped", n_steps)
            return True

        elif cmd == "img":
            image = self.env.get_camera_image()
            print("image shape =", image.shape, "dtype =", image.dtype)
            return True

        else:
            print(f"Unknown command: {cmd}")
            print("Type h for help.")
            return True

        obs = self.env.step(None, n_substeps=10, render=False)
        print("applied")
        print("arm =", np.round(self.current_arm, 5))
        print("gripper =", self.current_gripper)
        print("camera_pos =", np.round(obs["camera_pos"], 5))
        print("camera_euler =", np.round(obs["camera_euler"], 5))
        return True


def input_worker(commands: queue.Queue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            raw = input("rl-manual> ").strip()
        except (EOFError, KeyboardInterrupt):
            commands.put("q")
            return
        commands.put(raw)


def run_terminal(controller: ManualController) -> None:
    while True:
        try:
            raw = input("rl-manual> ").strip()
            if not controller.handle(raw):
                break
        except Exception as exc:
            print("Error:", exc)
        time.sleep(0.01)


def run_with_viewer(controller: ManualController, realtime: bool = True) -> None:
    command_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=input_worker, args=(command_queue, stop_event), daemon=True)
    thread.start()

    env = controller.env
    with env.launch_passive_viewer() as viewer:
        while viewer.is_running() and not stop_event.is_set():
            step_start = time.time()

            while True:
                try:
                    raw = command_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if not controller.handle(raw):
                        stop_event.set()
                        break
                except Exception as exc:
                    print("Error:", exc)

            env.step_simulation(n_substeps=1)
            viewer.sync()

            if realtime:
                elapsed = time.time() - step_start
                sleep_time = env.physics.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    stop_event.set()


def run_agent_loop(
    env: FrankaUmiCameraEnv,
    viewer: bool,
    realtime: bool,
    steps: int | None,
    n_substeps: int,
    render_for_agent: bool,
) -> None:
    """Run the future RL agent loop.

    This is the place that "refreshes" the simulation:
      - env.step(...) advances MuJoCo and returns the next observation.
      - viewer_handle.sync() refreshes the big MuJoCo window, if enabled.
    """
    agent = load_rl_agent()
    obs = env.reset(render=render_for_agent)
    step_count = 0

    if viewer:
        with env.launch_passive_viewer() as viewer_handle:
            while viewer_handle.is_running() and (steps is None or step_count < steps):
                step_start = time.time()
                obs = run_agent_step(env, agent, obs, n_substeps=n_substeps, render=render_for_agent)
                viewer_handle.sync()
                step_count += 1

                if realtime:
                    elapsed = time.time() - step_start
                    sleep_time = env.physics.model.opt.timestep * n_substeps - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
    else:
        while steps is None or step_count < steps:
            step_start = time.time()
            obs = run_agent_step(env, agent, obs, n_substeps=n_substeps, render=render_for_agent)
            step_count += 1

            if realtime:
                elapsed = time.time() - step_start
                sleep_time = env.physics.model.opt.timestep * n_substeps - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual controller that can replace an RL policy.")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer while accepting terminal input.")
    parser.add_argument("--no-realtime", action="store_true", help="Run as fast as possible in viewer mode.")
    parser.add_argument("--agent", action="store_true", help="Run the RL_AGENT_TODO policy loop instead of manual input.")
    parser.add_argument("--steps", type=int, default=None, help="Number of agent steps. Default: run forever.")
    parser.add_argument("--n-substeps", type=int, default=1, help="MuJoCo substeps per agent action.")
    parser.add_argument("--render-agent", action="store_true", help="Render camera images into obs for visual agents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = FrankaUmiCameraEnv()

    if args.agent:
        print("Running RL agent loop. Edit the RL_AGENT_TODO section in rl.py to plug in your model.")
        run_agent_loop(
            env=env,
            viewer=args.viewer,
            realtime=not args.no_realtime,
            steps=args.steps,
            n_substeps=args.n_substeps,
            render_for_agent=args.render_agent,
        )
        return

    controller = ManualController(env)

    print_help()
    print("Environment is ready. Type a command, then press Enter.")

    if args.viewer:
        print("MuJoCo viewer is open. Keep typing commands in this terminal.")
        run_with_viewer(controller, realtime=not args.no_realtime)
    else:
        run_terminal(controller)


if __name__ == "__main__":
    main()
