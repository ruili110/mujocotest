from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf


ROOT = Path(__file__).resolve().parent
MENAGERIE = ROOT / "mujoco_menagerie"

ENV_XML = MENAGERIE / "franka_emika_panda" / "plant_env.xml"
ARM_XML = MENAGERIE / "franka_emika_panda" / "panda_no_gripper.xml"
GRIPPER_XML = MENAGERIE / "umi_gripper" / "umi_gripper.xml"
MERGED_XML = ROOT / "merged_full_scene.xml"

INPUT_CAMERA_NAME = "input_camera"
INPUT_CAMERA_FULL_NAME = f"panda/umi_gripper/{INPUT_CAMERA_NAME}"
INPUT_CAMERA_MOUNT_NAME = "input_camera_mount"
GRIPPER_CAMERA_NAME = "gripper_gopro"

PANDA_JOINT_NAMES = tuple(f"panda/joint{i}" for i in range(1, 8))
PANDA_ACTUATOR_NAMES = tuple(f"panda/actuator{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = (
    "panda/umi_gripper/left_finger_joint",
    "panda/umi_gripper/right_finger_joint",
)
FINGER_ACTUATOR_NAME = "panda/umi_gripper/fingers_actuator"
CAMERA_JOINT_NAMES = (
    "panda/umi_gripper/input_camera_x",
    "panda/umi_gripper/input_camera_y",
    "panda/umi_gripper/input_camera_z",
)
CAMERA_ROT_JOINT_NAMES = (
    "panda/umi_gripper/input_camera_roll",
    "panda/umi_gripper/input_camera_pitch",
    "panda/umi_gripper/input_camera_yaw",
)
CAMERA_ACTUATOR_NAMES = (
    "panda/umi_gripper/input_camera_x_act",
    "panda/umi_gripper/input_camera_y_act",
    "panda/umi_gripper/input_camera_z_act",
)
CAMERA_ROT_ACTUATOR_NAMES = (
    "panda/umi_gripper/input_camera_roll_act",
    "panda/umi_gripper/input_camera_pitch_act",
    "panda/umi_gripper/input_camera_yaw_act",
)


@dataclass(frozen=True)
class CameraConfig:
    name: str = INPUT_CAMERA_NAME
    full_name: str = INPUT_CAMERA_FULL_NAME
    height: int = 480
    width: int = 640
    # Local UMI gripper frame. These values are exposed as MuJoCo controls.
    pos: tuple[float, float, float] = (0.0, -0.12249176, 0.05800533)
    euler: tuple[float, float, float] = (-0.6751, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    fovy: float = 80.0
    ctrlrange: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.25, 0.25),
        (-0.35, 0.15),
        (-0.15, 0.25),
    )
    rot_ctrlrange: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-1.57, 1.57),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )


def remove_umi_base_joints_and_actuators(gripper: mjcf.RootElement) -> None:
    """Fix the UMI gripper base so only the finger actuator remains controllable."""
    base_joint_names = {
        "gripper_joint_x",
        "gripper_joint_y",
        "gripper_joint_z",
        "gripper_joint_rx",
        "gripper_joint_ry",
        "gripper_joint_rz",
    }
    base_actuator_names = {
        "gripper_joint_x_act",
        "gripper_joint_y_act",
        "gripper_joint_z_act",
        "gripper_joint_rx_act",
        "gripper_joint_ry_act",
        "gripper_joint_rz_act",
    }

    for actuator in list(gripper.find_all("actuator")):
        if actuator.name in base_actuator_names:
            actuator.remove()

    for joint in list(gripper.find_all("joint")):
        if joint.name in base_joint_names:
            joint.remove()


def add_input_camera_to_umi_gripper(
    gripper: mjcf.RootElement,
    camera: CameraConfig = CameraConfig(),
) -> None:
    """Add a collisionless gripper camera whose xyz pose appears in MuJoCo controls."""
    if gripper.find("camera", camera.name) is not None:
        return

    gripper_bodies = list(gripper.worldbody.find_all("body"))
    if not gripper_bodies:
        raise RuntimeError("UMI gripper XML has no body to attach the input camera.")

    camera_mount = gripper_bodies[0].add("body", name=INPUT_CAMERA_MOUNT_NAME)
    camera_mount.add(
        "inertial",
        pos=[0.0, 0.0, 0.0],
        mass=0.001,
        diaginertia=[1e-6, 1e-6, 1e-6],
    )
    camera_mount.add(
        "joint",
        name="input_camera_x",
        type="slide",
        axis=[1.0, 0.0, 0.0],
        range=camera.ctrlrange[0],
        damping=1.0,
    )
    camera_mount.add(
        "joint",
        name="input_camera_y",
        type="slide",
        axis=[0.0, 1.0, 0.0],
        range=camera.ctrlrange[1],
        damping=1.0,
    )
    camera_mount.add(
        "joint",
        name="input_camera_z",
        type="slide",
        axis=[0.0, 0.0, 1.0],
        range=camera.ctrlrange[2],
        damping=1.0,
    )
    camera_mount.add(
        "joint",
        name="input_camera_roll",
        type="hinge",
        axis=[1.0, 0.0, 0.0],
        range=camera.rot_ctrlrange[0],
        damping=0.1,
    )
    camera_mount.add(
        "joint",
        name="input_camera_pitch",
        type="hinge",
        axis=[0.0, 1.0, 0.0],
        range=camera.rot_ctrlrange[1],
        damping=0.1,
    )
    camera_mount.add(
        "joint",
        name="input_camera_yaw",
        type="hinge",
        axis=[0.0, 0.0, 1.0],
        range=camera.rot_ctrlrange[2],
        damping=0.1,
    )
    camera_mount.add(
        "camera",
        name=camera.name,
        pos=[0.0, 0.0, 0.0],
        quat=camera.quat,
        mode="fixed",
        fovy=camera.fovy,
    )
    camera_mount.add(
        "geom",
        name="input_camera_marker",
        type="box",
        pos=[0.0, 0.0, 0.0],
        size=[0.018, 0.012, 0.012],
        rgba=[0.1, 0.35, 1.0, 0.75],
        contype=0,
        conaffinity=0,
        density=0,
        group=1,
    )
    camera_mount.add(
        "geom",
        name="input_camera_forward_marker",
        type="capsule",
        fromto=[0.0, 0.0, 0.0, 0.0, -0.04, 0.0],
        size=[0.003],
        rgba=[1.0, 0.2, 0.1, 0.85],
        contype=0,
        conaffinity=0,
        density=0,
        group=1,
    )

    actuators = gripper.actuator
    actuators.add(
        "position",
        name="input_camera_x_act",
        joint="input_camera_x",
        kp=1000,
        ctrlrange=camera.ctrlrange[0],
    )
    actuators.add(
        "position",
        name="input_camera_y_act",
        joint="input_camera_y",
        kp=1000,
        ctrlrange=camera.ctrlrange[1],
    )
    actuators.add(
        "position",
        name="input_camera_z_act",
        joint="input_camera_z",
        kp=1000,
        ctrlrange=camera.ctrlrange[2],
    )
    actuators.add(
        "position",
        name="input_camera_roll_act",
        joint="input_camera_roll",
        kp=100,
        ctrlrange=camera.rot_ctrlrange[0],
    )
    actuators.add(
        "position",
        name="input_camera_pitch_act",
        joint="input_camera_pitch",
        kp=100,
        ctrlrange=camera.rot_ctrlrange[1],
    )
    actuators.add(
        "position",
        name="input_camera_yaw_act",
        joint="input_camera_yaw",
        kp=100,
        ctrlrange=camera.rot_ctrlrange[2],
    )


def build_scene(
    camera: CameraConfig = CameraConfig(),
    export_xml: bool = True,
) -> mjcf.RootElement:
    env = mjcf.from_path(str(ENV_XML))
    arm = mjcf.from_path(str(ARM_XML))
    gripper = mjcf.from_path(str(GRIPPER_XML))

    remove_umi_base_joints_and_actuators(gripper)
    add_input_camera_to_umi_gripper(gripper, camera)

    link7 = arm.find("body", "link7")
    if link7 is None:
        raise RuntimeError("Could not find Panda body 'link7' in panda_no_gripper.xml.")

    mount_site = link7.add(
        "site",
        name="gripper_mount",
        pos=[0.0, 0.0, 0.3],
        euler=[0.0, np.pi, 2.6],
    )
    mount_site.attach(gripper)

    arm_base_site = env.worldbody.add("site", name="arm_base", pos=[0.0, 0.0, 0.0])
    arm_base_site.attach(arm)

    if export_xml:
        MERGED_XML.write_text(env.to_xml_string(), encoding="utf-8")

    return env


def set_initial_pose(
    physics: mjcf.Physics,
    gripper_opening: float = 0.03,
    camera: CameraConfig = CameraConfig(),
) -> None:
    home_arm = {
        # Terminal pose from "Demeter_Exp1 - new pos.task" (Pose 2).
        "panda/joint1": -0.01617866471870486,
        "panda/joint2": -0.3351507979411188,
        "panda/joint3": 0.11335772886207839,
        "panda/joint4": -1.3269075321321429,
        "panda/joint5": 0.021510896893726657,
        "panda/joint6": 0.7344821652580072,
        "panda/joint7": 0.8923453157986305,
    }

    for joint_name, value in home_arm.items():
        physics.named.data.qpos[joint_name] = value

    for actuator_name, joint_name in zip(PANDA_ACTUATOR_NAMES, PANDA_JOINT_NAMES):
        joint_position = physics.named.data.qpos[joint_name].item()
        physics.named.data.ctrl[actuator_name] = joint_position

    for joint_name in FINGER_JOINT_NAMES:
        physics.named.data.qpos[joint_name] = gripper_opening
    physics.named.data.ctrl[FINGER_ACTUATOR_NAME] = gripper_opening

    for joint_name, actuator_name, value in zip(
        CAMERA_JOINT_NAMES,
        CAMERA_ACTUATOR_NAMES,
        camera.pos,
    ):
        physics.named.data.qpos[joint_name] = value
        physics.named.data.ctrl[actuator_name] = value
    for joint_name, actuator_name, value in zip(
        CAMERA_ROT_JOINT_NAMES,
        CAMERA_ROT_ACTUATOR_NAMES,
        camera.euler,
    ):
        physics.named.data.qpos[joint_name] = value
        physics.named.data.ctrl[actuator_name] = value

    physics.data.qvel[:] = 0.0
    physics.forward()


class FrankaUmiCameraEnv:
    """Direct in-process interface for policy, diffusion-model, and RL experiments.

    Action formats accepted by step/apply_action:
      - ndarray/list with length model.nu: full actuator ctrl vector.
      - mapping with "arm": 7 Panda targets and optional "gripper" scalar opening.
      - mapping with "camera_pos" xyz and/or "camera_euler" roll/pitch/yaw.
      - mapping from actuator name to scalar target.
    """

    def __init__(
        self,
        camera: CameraConfig = CameraConfig(),
        export_xml: bool = True,
        gripper_opening: float = 0.03,
    ) -> None:
        self.camera = camera
        self.mjcf_model = build_scene(camera=camera, export_xml=export_xml)
        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_model)
        set_initial_pose(self.physics, gripper_opening=gripper_opening, camera=camera)

    @property
    def model(self) -> mujoco.MjModel:
        return self.physics.model.ptr

    @property
    def data(self) -> mujoco.MjData:
        return self.physics.data.ptr

    @property
    def actuator_names(self) -> list[str]:
        return [self.physics.model.id2name(i, "actuator") for i in range(self.physics.model.nu)]

    def reset(self, gripper_opening: float = 0.03, render: bool = True) -> dict[str, np.ndarray]:
        self.physics.reset()
        set_initial_pose(self.physics, gripper_opening=gripper_opening, camera=self.camera)
        return self.get_observation(render=render)

    def set_control(
        self,
        arm: Sequence[float] | None = None,
        gripper: float | None = None,
        camera_pos: Sequence[float] | None = None,
        camera_euler: Sequence[float] | None = None,
        ctrl: Sequence[float] | None = None,
    ) -> None:
        """Set actuator targets without stepping the simulator."""
        action: dict[str, object] = {}
        if ctrl is not None:
            action["ctrl"] = ctrl
        if arm is not None:
            action["arm"] = arm
        if gripper is not None:
            action["gripper"] = gripper
        if camera_pos is not None:
            action["camera_pos"] = camera_pos
        if camera_euler is not None:
            action["camera_euler"] = camera_euler
        if action:
            self.apply_action(action)

    def apply_action(self, action: Sequence[float] | Mapping[str, object]) -> None:
        if isinstance(action, Mapping):
            self._apply_mapping_action(action)
        else:
            ctrl = np.asarray(action, dtype=float)
            if ctrl.shape != (self.physics.model.nu,):
                raise ValueError(f"Expected full ctrl shape {(self.physics.model.nu,)}, got {ctrl.shape}.")
            self.physics.data.ctrl[:] = self._clip_ctrl(ctrl)

    def step(
        self,
        action: Sequence[float] | Mapping[str, object] | None = None,
        n_substeps: int = 1,
        render: bool = True,
    ) -> dict[str, np.ndarray]:
        if action is not None:
            self.apply_action(action)

        for _ in range(n_substeps):
            self.physics.step()

        return self.get_observation(render=render)

    def step_simulation(self, n_substeps: int = 1) -> None:
        """Advance physics without rendering or allocating an observation dict."""
        for _ in range(n_substeps):
            self.physics.step()

    def get_camera_image(
        self,
        camera_id: str | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        return self.physics.render(
            height=height or self.camera.height,
            width=width or self.camera.width,
            camera_id=camera_id or self.camera.full_name,
        )

    def get_observation(self, render: bool = True) -> dict[str, np.ndarray]:
        obs = {
            "qpos": self.physics.data.qpos.copy(),
            "qvel": self.physics.data.qvel.copy(),
            "ctrl": self.physics.data.ctrl.copy(),
            "camera_pos": self.get_camera_pos(),
            "camera_euler": self.get_camera_euler(),
        }
        if render:
            obs["image"] = self.get_camera_image()
        return obs

    def observe(self, render: bool = True) -> dict[str, np.ndarray]:
        """Alias for get_observation(), useful in policy loops."""
        return self.get_observation(render=render)

    def get_camera_pos(self) -> np.ndarray:
        return np.array([self.physics.named.data.qpos[name].item() for name in CAMERA_JOINT_NAMES], dtype=float)

    def get_camera_euler(self) -> np.ndarray:
        return np.array(
            [self.physics.named.data.qpos[name].item() for name in CAMERA_ROT_JOINT_NAMES],
            dtype=float,
        )

    def launch_viewer(self) -> None:
        self.run_viewer()

    def launch_passive_viewer(self) -> mujoco.viewer.Handle:
        return mujoco.viewer.launch_passive(self.model, self.data)

    def run_viewer(
        self,
        realtime: bool = True,
        show_camera: bool = True,
        camera_window_name: str = "UMI input_camera",
    ) -> None:
        """Run the MuJoCo viewer and optionally show the gripper camera image."""
        cv2 = None
        if show_camera:
            try:
                import cv2 as cv2_module

                cv2 = cv2_module
            except ImportError:
                print("OpenCV is not installed, so only the MuJoCo viewer will be shown.")
                print("Install it with: pip install opencv-python")

        with self.launch_passive_viewer() as viewer:
            while viewer.is_running():
                step_start = time.time()
                self.physics.step()
                viewer.sync()

                if cv2 is not None:
                    rgb = self.get_camera_image()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(camera_window_name, bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if realtime:
                    elapsed = time.time() - step_start
                    sleep_time = self.physics.model.opt.timestep - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        if cv2 is not None:
            cv2.destroyWindow(camera_window_name)

        print(f"Final input_camera xyz: {self._format_float_tuple(self.get_camera_pos())}")
        print(f"Final input_camera roll/pitch/yaw: {self._format_float_tuple(self.get_camera_euler())}")

    def run_policy(
        self,
        policy: Callable[[dict[str, np.ndarray]], Sequence[float] | Mapping[str, object] | None],
        n_steps: int | None = None,
        n_substeps: int = 1,
        render: bool = True,
        realtime: bool = False,
    ) -> None:
        """Run a user policy in-process.

        The policy receives obs and returns an action dict/list accepted by step().
        Use render=False for fast RL rollouts and render=True when the policy needs images.
        """
        obs = self.observe(render=render)
        step_count = 0
        while n_steps is None or step_count < n_steps:
            step_start = time.time()
            action = policy(obs)
            obs = self.step(action, n_substeps=n_substeps, render=render)
            step_count += 1

            if realtime:
                elapsed = time.time() - step_start
                sleep_time = self.physics.model.opt.timestep * n_substeps - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _apply_mapping_action(self, action: Mapping[str, object]) -> None:
        if "ctrl" in action:
            self.apply_action(action["ctrl"])  # type: ignore[arg-type]

        if "arm" in action:
            arm = np.asarray(action["arm"], dtype=float)
            if arm.shape != (7,):
                raise ValueError(f"Expected 'arm' action shape (7,), got {arm.shape}.")
            for name, value in zip(PANDA_ACTUATOR_NAMES, arm):
                self.physics.named.data.ctrl[name] = value

        if "gripper" in action:
            self.physics.named.data.ctrl[FINGER_ACTUATOR_NAME] = float(action["gripper"])

        if "camera_pos" in action:
            camera_pos = np.asarray(action["camera_pos"], dtype=float)
            if camera_pos.shape != (3,):
                raise ValueError(f"Expected 'camera_pos' action shape (3,), got {camera_pos.shape}.")
            for name, value in zip(CAMERA_ACTUATOR_NAMES, camera_pos):
                self.physics.named.data.ctrl[name] = value

        if "camera_euler" in action:
            camera_euler = np.asarray(action["camera_euler"], dtype=float)
            if camera_euler.shape != (3,):
                raise ValueError(f"Expected 'camera_euler' action shape (3,), got {camera_euler.shape}.")
            for name, value in zip(CAMERA_ROT_ACTUATOR_NAMES, camera_euler):
                self.physics.named.data.ctrl[name] = value

        for name, value in action.items():
            if name in {"ctrl", "arm", "gripper", "camera_pos", "camera_euler"}:
                continue
            if name not in self.actuator_names:
                raise KeyError(f"Unknown actuator '{name}'. Known actuators: {self.actuator_names}")
            self.physics.named.data.ctrl[name] = float(value)

        self.physics.data.ctrl[:] = self._clip_ctrl(self.physics.data.ctrl)

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        ranges = self.physics.model.actuator_ctrlrange
        limited = self.physics.model.actuator_ctrllimited.astype(bool)
        clipped = ctrl.copy()
        clipped[limited] = np.clip(clipped[limited], ranges[limited, 0], ranges[limited, 1])
        return clipped

    @staticmethod
    def _format_float_tuple(values: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in values)


def print_model_info(env: FrankaUmiCameraEnv) -> None:
    print("\n===== Joint Names =====")
    for i in range(env.physics.model.njnt):
        print(i, env.physics.model.id2name(i, "joint"))

    print("\n===== Actuator Names =====")
    for i, name in enumerate(env.actuator_names):
        print(i, name)

    print("\n===== Camera Names =====")
    for i in range(env.physics.model.ncam):
        print(i, env.physics.model.id2name(i, "camera"))

    print("\nnq =", env.physics.model.nq)
    print("nv =", env.physics.model.nv)
    print("nu =", env.physics.model.nu)


def build_and_run_scene() -> FrankaUmiCameraEnv:
    env = FrankaUmiCameraEnv()
    print_model_info(env)
    print(f"\nCamera input shape: ({env.camera.height}, {env.camera.width}, 3)")
    print(f"Camera pose controls: xyz={env.camera.pos}, roll/pitch/yaw={env.camera.euler}")
    print(f"Merged XML exported to: {MERGED_XML}")
    env.run_viewer(show_camera=True)
    return env


if __name__ == "__main__":
    build_and_run_scene()
