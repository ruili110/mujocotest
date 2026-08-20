from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.mujoco import wrapper


ROOT = Path(__file__).resolve().parent
SHARED_CONFIG = ROOT / "diffusion_mujoco_config.json"
MENAGERIE = ROOT / "mujoco_menagerie"

ENV_XML = MENAGERIE / "franka_emika_panda" / "plant_env.xml"
ARM_XML = MENAGERIE / "franka_emika_panda" / "panda_no_gripper.xml"
GRIPPER_XML = MENAGERIE / "umi_gripper" / "umi_gripper.xml"
MERGED_XML = ROOT / "merged_full_scene.xml"

INPUT_CAMERA_NAME = "input_camera"
# The policy and the preview both use this adjustable camera. Its six position
# actuators appear in MuJoCo's Control panel, so calibration changes are visible
# immediately and affect the exact image sent to the model.
INPUT_CAMERA_FULL_NAME = "panda/umi_gripper/input_camera"
INPUT_CAMERA_MOUNT_NAME = "input_camera_mount"

# Mechanical adapter pose from Panda link7 to the UMI-gripper base.
# Calibrated link7-to-gripper translation; panda/joint7 remains unchanged.
GRIPPER_MOUNT_POS = (0.005, -0.010, 0.23)
# Fixed mechanical-adapter orientation plus the calibrated live-control offset.
GRIPPER_MOUNT_EULER = (0.0, np.pi, 2.2923453157986306)
GRIPPER_MOUNT_CONTROL_EULER = (0.0, 0.0, -np.pi / 2.0)

# Real payload configuration used by farm/franka_panda/config/panda_config.yaml.
# The existing UMI XML contributes 0.274372022 kg from its two moving fingers,
# GoPro collision mesh and simulated camera mount. A fixed correction body adds
# only the missing mass, so the complete compiled payload is exactly 1 kg while
# preserving the finger contact dynamics.
FARM_GRIPPER_TOTAL_MASS = 1.0
FARM_GRIPPER_COM = (0.065, -0.056, 0.028)
FARM_GRIPPER_MASS_CORRECTION = 0.7256279779955341
FARM_GRIPPER_CORRECTION_COM = (0.08672300, -0.09956538, -0.01512873)
FARM_GRIPPER_INERTIA_APPROX = (1e-4, 1e-4, 1e-4)

# Names used by the live scene-pose sliders. These sliders are interpreted only
# by run_viewer(); normal policy rollouts keep the pot and plant freely dynamic.
LIVE_POSE_OBJECTS = ("support_box", "pot", "plant", "gripper_mount")
LIVE_POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
LIVE_POSE_ACTUATOR_NAMES = tuple(
    f"scene_{object_name}_{axis}"
    for object_name in LIVE_POSE_OBJECTS
    for axis in LIVE_POSE_AXES
)

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
    # Standalone simulated-camera stream. MuJoCo directly stores only vertical
    # FOV; width=822 at height=480 makes fovy=58 produce hfov≈87.02 degrees.
    # This intentionally does not use the camera from the UMI XML.
    height: int = 480
    width: int = 822
    # Camera pose in the local UMI-gripper frame, measured in metres/radians.
    # pos=(x,y,z); euler=(roll,pitch,yaw). These are the initial values shown by
    # the six camera sliders in MuJoCo's Control panel.
    # Latest Control-panel calibration, in metres. Preserve the measured values
    # exactly, including the sub-micrometre x residual.
    pos: tuple[float, float, float] = (
        -5.935802852939265e-07,
        -0.009,
        0.026,
    )
    # Latest Control-panel calibration (roll/pitch/yaw, radians).
    euler: tuple[float, float, float] = (
        -0.518,
        0.0,
        0.0,
    )
    # Original standalone-camera optical orientation, quaternion=(w,x,y,z).
    # The mount's live roll/pitch/yaw joints are applied on top of this value.
    quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    # Requested optical field of view: horizontal≈87°, vertical=58°.
    fovy: float = 58.0
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


@dataclass(frozen=True)
class ThirdPersonCameraConfig:
    """Initial MuJoCo viewer camera; separate from the policy input camera."""

    # World-space point at which the third-person camera looks (metres).
    lookat: tuple[float, float, float] = (0.25, 0.0, 0.55)
    # Axis-aligned front view: azimuth=180 avoids the misleading diagonal view.
    azimuth: float = 180.0
    elevation: float = -15.0
    distance: float = 1.55


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


def make_umi_gripper_transparent(gripper: mjcf.RootElement) -> None:
    """Hide UMI geometry without disabling its joints, mass, or collisions.

    Setting only the render alpha to zero keeps every collision mesh active, so
    grasp physics is unchanged while the wrist camera gets an unobstructed view.
    This includes the finger meshes, ArUco stickers, camera shell, and the small
    input-camera pose markers added below.
    """
    for geom in gripper.find_all("geom"):
        geom.rgba = [1.0, 1.0, 1.0, 0.0]


def put_umi_gripper_in_viewer_only_group(gripper: mjcf.RootElement) -> None:
    """Show UMI geometry in the viewer while excluding it from policy RGB.

    Geom groups affect rendering only; mass, contacts and actuation remain
    unchanged. Group 4 is enabled in the viewer and disabled for input_camera.
    """
    for geom in gripper.find_all("geom"):
        geom.group = 4


def add_farm_payload_mass_correction(gripper: mjcf.RootElement) -> None:
    """Match farm's 1 kg gripper payload without double-counting UMI parts."""
    correction = gripper.worldbody.add(
        "body",
        name="farm_payload_mass_correction",
        pos=FARM_GRIPPER_CORRECTION_COM,
    )
    correction.add(
        "inertial",
        pos=(0.0, 0.0, 0.0),
        mass=FARM_GRIPPER_MASS_CORRECTION,
        diaginertia=FARM_GRIPPER_INERTIA_APPROX,
    )


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


def add_live_scene_pose_controls(env: mjcf.RootElement) -> None:
    """Add Control-panel sliders used by the interactive calibration viewer.

    The sliders drive tiny invisible dummy joints. ``run_viewer`` reads their
    actuator control values and applies them to the requested object poses. This
    keeps the pot/plant free joints physically correct during policy rollouts,
    where this viewer-only pose override is not executed.
    """
    for object_name in LIVE_POSE_OBJECTS:
        for axis_index, axis_name in enumerate(LIVE_POSE_AXES):
            is_rotation = axis_index >= 3
            if is_rotation:
                control_range = [-np.pi, np.pi]
            elif object_name == "gripper_mount" and axis_name in {"x", "y"}:
                # Keep lateral adapter calibration close to the link7 axis.
                control_range = [-0.10, 0.10]
            elif object_name == "gripper_mount" and axis_name == "z":
                # Prevent the base from being pulled deeply into link7 while
                # still allowing +/- several centimetres of useful tuning.
                control_range = [0.18, 0.35]
            else:
                control_range = [-1.0, 1.0]
            dummy = env.worldbody.add(
                "body",
                name=f"scene_{object_name}_{axis_name}_control_body",
                pos=[0.0, 0.0, -2.0],
                gravcomp=1.0,
            )
            dummy.add(
                "inertial",
                pos=[0.0, 0.0, 0.0],
                mass=0.001,
                diaginertia=[1e-6, 1e-6, 1e-6],
            )
            joint_name = f"scene_{object_name}_{axis_name}_control_joint"
            dummy.add(
                "joint",
                name=joint_name,
                type="hinge" if is_rotation else "slide",
                axis=[1.0, 0.0, 0.0],
                damping=1.0,
                range=control_range,
            )
            env.actuator.add(
                "position",
                name=f"scene_{object_name}_{axis_name}",
                joint=joint_name,
                kp=1.0,
                ctrlrange=control_range,
            )


def build_scene(
    camera: CameraConfig = CameraConfig(),
    export_xml: bool = True,
) -> mjcf.RootElement:
    env = mjcf.from_path(str(ENV_XML))
    arm = mjcf.from_path(str(ARM_XML))
    gripper = mjcf.from_path(str(GRIPPER_XML))

    # The old workflow uses only our adjustable standalone input_camera. Remove
    # the UMI XML camera sensor entirely so it cannot be selected accidentally;
    # this does not remove the visible camera mesh from the gripper model.
    built_in_camera = gripper.find("camera", "gripper_gopro")
    if built_in_camera is not None:
        built_in_camera.remove()

    remove_umi_base_joints_and_actuators(gripper)
    add_input_camera_to_umi_gripper(gripper, camera)
    add_farm_payload_mass_correction(gripper)
    # Preserve the UMI XML's original render groups: visual meshes are group 2
    # and collision meshes are group 3. Render options below show the former
    # and hide the latter without changing collision physics.

    link7 = arm.find("body", "link7")
    if link7 is None:
        raise RuntimeError("Could not find Panda body 'link7' in panda_no_gripper.xml.")

    mount_site = link7.add(
        "site",
        name="gripper_mount",
        pos=GRIPPER_MOUNT_POS,
        # Compose the fixed adapter with the saved Control-panel rotation.
        # Panda joint7 remains exactly equal to the real-robot task.
        quat=quat_multiply(
            euler_xyz_to_quat(GRIPPER_MOUNT_EULER),
            euler_xyz_to_quat(GRIPPER_MOUNT_CONTROL_EULER),
        ),
    )
    mount_site.attach(gripper)

    arm_base_site = env.worldbody.add("site", name="arm_base", pos=[0.0, 0.0, 0.0])
    arm_base_site.attach(arm)

    # Viewer-only position/rotation sliders for the black box, pot, and plant.
    add_live_scene_pose_controls(env)

    if export_xml:
        MERGED_XML.write_text(env.to_xml_string(), encoding="utf-8")

    return env


def set_initial_pose(
    physics: mjcf.Physics,
    gripper_opening: float = 0,
    camera: CameraConfig = CameraConfig(),
) -> None:
    home_arm = {
        # Final pose from "Demeter_Exp1 - new pos.task" (Pose 2).
        # Recorded TCP xyz = (0.3059878220, 0.0487312520, 0.8111367658) m.
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
        physics.named.data.ctrl[actuator_name] = np.asarray(
            physics.named.data.qpos[joint_name]
        ).item()

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


def euler_xyz_to_quat(euler: Sequence[float]) -> np.ndarray:
    """Convert roll/pitch/yaw radians to MuJoCo's (w, x, y, z) quaternion."""
    roll, pitch, yaw = np.asarray(euler, dtype=float) * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def quat_to_euler_xyz(quat: Sequence[float]) -> np.ndarray:
    """Convert a MuJoCo (w, x, y, z) quaternion to roll/pitch/yaw radians."""
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    matrix = matrix.reshape(3, 3)
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def quat_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    """Compose two MuJoCo-order quaternions as ``left * right``."""
    lw, lx, ly, lz = np.asarray(left, dtype=float)
    rw, rx, ry, rz = np.asarray(right, dtype=float)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


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
        third_person_camera: ThirdPersonCameraConfig = ThirdPersonCameraConfig(),
        export_xml: bool = True,
        gripper_opening: float = 0.0,
    ) -> None:
        self.camera = camera
        self.third_person_camera = third_person_camera
        self.mjcf_model = build_scene(camera=camera, export_xml=export_xml)
        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_model)
        # Hide raw collision meshes (group 3) and colored collision-debug
        # shapes (group 5) from policy images; contact physics stays enabled.
        self.camera_scene_option = wrapper.MjvOption()
        self.camera_scene_option.geomgroup[3] = 0
        self.camera_scene_option.geomgroup[5] = 0
        set_initial_pose(self.physics, gripper_opening=gripper_opening, camera=camera)
        self._initialize_live_pose_controls()

    @property
    def model(self) -> mujoco.MjModel:
        return self.physics.model.ptr

    @property
    def data(self) -> mujoco.MjData:
        return self.physics.data.ptr

    @property
    def actuator_names(self) -> list[str]:
        return [self.physics.model.id2name(i, "actuator") for i in range(self.physics.model.nu)]

    def reset(self, gripper_opening: float = 0.0, render: bool = True) -> dict[str, np.ndarray]:
        self.physics.reset()
        set_initial_pose(self.physics, gripper_opening=gripper_opening, camera=self.camera)
        self._initialize_live_pose_controls()
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
            scene_option=self.camera_scene_option,
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
        return np.array(
            [np.asarray(self.physics.named.data.qpos[name]).item() for name in CAMERA_JOINT_NAMES],
            dtype=float,
        )

    def get_camera_euler(self) -> np.ndarray:
        return np.array(
            [np.asarray(self.physics.named.data.qpos[name]).item() for name in CAMERA_ROT_JOINT_NAMES],
            dtype=float,
        )

    def _get_scene_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return world xyz and roll/pitch/yaw for one live-adjustable object."""
        if object_name == "gripper_mount":
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "panda/umi_gripper/",
            )
            return (
                self.model.body_pos[body_id].copy(),
                # Saved Control-panel rotation relative to the fixed adapter.
                np.asarray(GRIPPER_MOUNT_CONTROL_EULER, dtype=float),
            )

        if object_name == "support_box":
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "support_box_body",
            )
            return (
                self.model.body_pos[body_id].copy(),
                quat_to_euler_xyz(self.model.body_quat[body_id]),
            )

        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{object_name}_freejoint",
        )
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        free_pose = self.data.qpos[qpos_address : qpos_address + 7]
        return free_pose[:3].copy(), quat_to_euler_xyz(free_pose[3:7])

    def _initialize_live_pose_controls(self) -> None:
        """Seed all scene sliders from the XML poses instead of zero."""
        for object_name in LIVE_POSE_OBJECTS:
            position, euler = self._get_scene_object_pose(object_name)
            for axis_name, value in zip(LIVE_POSE_AXES, np.concatenate([position, euler])):
                self.physics.named.data.ctrl[f"scene_{object_name}_{axis_name}"] = value

    def _apply_live_pose_controls(self) -> None:
        """Apply Control-panel object sliders during direct interactive viewing."""
        for object_name in LIVE_POSE_OBJECTS:
            values = np.array(
                [
                    np.asarray(
                        self.physics.named.data.ctrl[f"scene_{object_name}_{axis_name}"]
                    ).item()
                    for axis_name in LIVE_POSE_AXES
                ],
                dtype=float,
            )
            position = values[:3]
            quat = euler_xyz_to_quat(values[3:])

            if object_name == "gripper_mount":
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "panda/umi_gripper/",
                )
                self.model.body_pos[body_id] = position
                base_quat = euler_xyz_to_quat(GRIPPER_MOUNT_EULER)
                self.model.body_quat[body_id] = quat_multiply(base_quat, quat)
                continue

            if object_name == "support_box":
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "support_box_body",
                )
                self.model.body_pos[body_id] = position
                self.model.body_quat[body_id] = quat
                continue

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{object_name}_freejoint",
            )
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_address : qpos_address + 3] = position
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = quat
            self.data.qvel[dof_address : dof_address + 6] = 0.0

        self.physics.forward()

    def get_live_scene_control_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read the exact xyz/rpy targets currently displayed in Control."""
        if object_name not in LIVE_POSE_OBJECTS:
            raise KeyError(f"Unknown live-pose object: {object_name}")
        values = np.array(
            [
                np.asarray(
                    self.physics.named.data.ctrl[f"scene_{object_name}_{axis_name}"]
                ).item()
                for axis_name in LIVE_POSE_AXES
            ],
            dtype=float,
        )
        return values[:3], values[3:]

    def launch_viewer(self) -> None:
        self.run_viewer()

    def launch_passive_viewer(self) -> mujoco.viewer.Handle:
        return mujoco.viewer.launch_passive(self.model, self.data)

    def run_viewer(
        self,
        realtime: bool = True,
        show_camera: bool = True,
        camera_window_name: str = "Standalone policy input_camera",
    ) -> None:
        """Run MuJoCo with live xyz/rpy sliders in the viewer's Control panel."""
        cv2 = None
        if show_camera:
            try:
                import cv2 as cv2_module

                cv2 = cv2_module
            except ImportError:
                print("OpenCV is not installed, so only the MuJoCo viewer will be shown.")
                print("Install it with: pip install opencv-python")

        print("\nLive pose calibration: open the MuJoCo viewer's Control panel.")
        print("  scene_support_box_[x/y/z/roll/pitch/yaw] : black box world pose")
        print("  scene_pot_[x/y/z/roll/pitch/yaw]         : pot world pose")
        print("  scene_plant_[x/y/z/roll/pitch/yaw]       : plant world pose")
        print("  scene_gripper_mount_[x/y/z]              : link7-to-UMI adapter position")
        print("  scene_gripper_mount_[roll/pitch/yaw]     : adapter rotation offset")
        print("  input_camera_[x/y/z/roll/pitch/yaw]_act : camera pose in gripper frame")
        print("Units: translation=metres, rotation=radians. Press Q in the camera window to finish.\n")

        with self.launch_passive_viewer() as viewer:
            # Set a deterministic, world-upright front view. Policy-camera
            # resolution, FOV, pose, and input are independent of this view.
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = self.third_person_camera.lookat
            viewer.cam.azimuth = self.third_person_camera.azimuth
            viewer.cam.elevation = self.third_person_camera.elevation
            viewer.cam.distance = self.third_person_camera.distance
            # Show all ordinary visual geometry and the main plant collision
            # capsule (5), while hiding every other collision shape (3).
            # The policy camera independently hides both groups 3 and 5.
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[5] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
            viewer.sync()
            while viewer.is_running():
                step_start = time.time()
                self.physics.step()
                # Direct new2.py execution is calibration mode: the object pose
                # sliders are applied after simulation and before both renderers.
                self._apply_live_pose_controls()
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
        for object_name in LIVE_POSE_OBJECTS:
            position, euler = self.get_live_scene_control_pose(object_name)
            print(f"Final {object_name} xyz: {self._format_float_tuple(position)}")
            print(f"Final {object_name} roll/pitch/yaw: {self._format_float_tuple(euler)}")

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
    # Direct new2.py execution reads the same camera and scene configuration as
    # diffusion_mujoco.py, so Control starts from the JSON values shown to users.
    with SHARED_CONFIG.open("r", encoding="utf-8") as file:
        shared = json.load(file)
    camera_values = shared["camera"]
    third_values = camera_values["third_person"]
    camera = CameraConfig(
        width=int(camera_values["width"]),
        height=int(camera_values["height"]),
        fovy=float(camera_values["fovy_degrees"]),
        pos=tuple(float(value) for value in camera_values["xyz"]),
        euler=tuple(float(value) for value in camera_values["rpy"]),
    )
    third_person = ThirdPersonCameraConfig(
        lookat=tuple(float(value) for value in third_values["lookat"]),
        azimuth=float(third_values["azimuth_degrees"]),
        elevation=float(third_values["elevation_degrees"]),
        distance=float(third_values["distance"]),
    )
    env = FrankaUmiCameraEnv(
        camera=camera,
        third_person_camera=third_person,
        gripper_opening=float(shared["gripper"]["initial_joint"]),
    )
    for object_name in LIVE_POSE_OBJECTS:
        pose = shared["scene"][object_name]
        values = [*pose["xyz"], *pose["rpy"]]
        for axis_name, value in zip(LIVE_POSE_AXES, values):
            env.physics.named.data.ctrl[f"scene_{object_name}_{axis_name}"] = float(value)
    env._apply_live_pose_controls()
    print_model_info(env)
    print(f"\nCamera input shape: ({env.camera.height}, {env.camera.width}, 3)")
    print(f"Camera pose controls: xyz={env.camera.pos}, roll/pitch/yaw={env.camera.euler}")
    print(
        "Third-person viewer: "
        f"lookat={env.third_person_camera.lookat}, "
        f"azimuth={env.third_person_camera.azimuth}, "
        f"elevation={env.third_person_camera.elevation}, "
        f"distance={env.third_person_camera.distance}"
    )
    print(f"Merged XML exported to: {MERGED_XML}")
    env.run_viewer(show_camera=True)
    return env


if __name__ == "__main__":
    build_and_run_scene()
