from __future__ import annotations

"""Minimal testdemeter diffusion-policy runner for the MuJoCo Franka scene."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from new2 import (
    FINGER_JOINT_NAMES,
    GRIPPER_MOUNT_EULER,
    PANDA_JOINT_NAMES,
    CameraConfig,
    FrankaUmiCameraEnv,
    ThirdPersonCameraConfig,
    euler_xyz_to_quat,
    quat_multiply,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "diffusion_mujoco_config.json"
REFERENCE_CONFIG_MODULE = "lerobot.common.policies.diffusion.configuration_diffusion"
REFERENCE_MODEL_MODULE = "lerobot.common.policies.diffusion.modeling_diffusion"


def rotation_to_feature(rot: Rotation) -> np.ndarray:
    matrix = rot.inv().as_matrix()
    return matrix.reshape((*matrix.shape[:-2], -1))[..., 3:]


def feature_to_rotation(feature: np.ndarray) -> Rotation:
    z_raw = feature[..., 3:]
    z_norm = np.linalg.norm(z_raw, axis=-1, keepdims=True)
    assert np.all(z_norm > 0)
    z_axis = z_raw / z_norm
    y_raw = feature[..., :3] - (z_axis * feature[..., :3]).sum(
        -1, keepdims=True
    ) * z_axis
    y_norm = np.linalg.norm(y_raw, axis=-1, keepdims=True)
    assert np.all(y_norm > 0)
    y_axis = y_raw / y_norm
    x_axis = np.cross(y_axis, z_axis)
    return Rotation.from_matrix(np.stack([x_axis, y_axis, z_axis], axis=-1))


def vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"{name} must contain {length} values, got {result.shape}")
    return result


def load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_policy(model: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    mode = str(model["mode"])
    model_path = Path(model["checkpoints"][mode]).resolve()
    lerobot_source = Path(model["lerobot_source"]).resolve()
    reference_dir = Path(model["reference_code_dir"]).resolve()
    config_file = reference_dir / "configuration_diffusion.py"
    modeling_file = reference_dir / "modeling_diffusion (1).py"
    for path in (
        model_path / "config.json",
        model_path / "model.safetensors",
        config_file,
        modeling_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(lerobot_source))

    # The supplied model imports an OT helper that is absent from the supplied
    # files and is not used by select_action. The stub fails if training calls it.
    package_name = "lerobot.common.policies.diffusion"
    package = ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    ot_module = ModuleType(f"{package_name}.ot_utils")

    def missing_ot_loss(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("testdemeter did not provide the training-only ot_utils.py")

    ot_module.compute_ot_loss = missing_ot_loss  # type: ignore[attr-defined]
    sys.modules[ot_module.__name__] = ot_module

    load_module(REFERENCE_CONFIG_MODULE, config_file)
    modeling = load_module(REFERENCE_MODEL_MODULE, modeling_file)
    policy = modeling.DiffusionPolicy.from_pretrained(model_path)
    policy.reset()

    import torch

    device = next(policy.parameters()).device
    if policy.config.use_green_grass_encoder:
        encoder_mode = "green_center"
    elif policy.config.use_rgb_encoder:
        encoder_mode = "resnet18"
    else:
        encoder_mode = "other"
    return policy, torch, device, encoder_mode


class TestDemeterPolicy:
    def __init__(self, model: dict[str, Any]) -> None:
        self.policy, self.torch, self.device, self.encoder_mode = (
            load_reference_policy(model)
        )

    def select_action(
        self,
        gripper_width: float,
        rgb_batch: np.ndarray,
        ee_position: np.ndarray,
        ee_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.concatenate(
            [
                [gripper_width],
                ee_position.astype(np.float32),
                rotation_to_feature(Rotation.from_quat(ee_quaternion)).astype(
                    np.float32
                ),
            ]
        ).astype(np.float32)
        observation = {
            "observation.state": self.torch.from_numpy(state)
            .unsqueeze(0)
            .to(self.device, non_blocking=True),
            "observation.image.realsense": self.torch.from_numpy(rgb_batch)
            .float()
            .div(255)
            .permute(0, 3, 1, 2)
            .to(self.device, non_blocking=True),
        }
        with self.torch.inference_mode():
            action = self.policy.select_action(observation)
        return state, action.squeeze(0).to("cpu").numpy()


class PandaMoveAbsIK:
    def __init__(self, env: FrankaUmiCameraEnv, config: dict[str, Any]) -> None:
        self.env = env
        self.model = env.model
        self.link7_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "panda/link7")
        self.ee_site_id = self._id(
            mujoco.mjtObj.mjOBJ_SITE, "panda/gripper_mount"
        )
        self.joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in PANDA_JOINT_NAMES]
        )
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids].copy()
        self.ranges = self.model.jnt_range[self.joint_ids].copy()
        self.data = mujoco.MjData(self.model)
        self.max_nfev = int(config["max_function_evaluations"])
        self.ftol = float(config["ftol"])
        self.xtol = float(config["xtol"])
        self.gtol = float(config["gtol"])
        self.accept_position_error = float(config["accept_position_error_metres"])
        self.accept_rotation_error = float(config["accept_rotation_error_radians"])
        # Farm continuous-IK behavior: initialize from the measured joints once,
        # then keep the previous successful IK branch for the next model target.
        self.initial_solution = self.env.data.qpos[self.qpos_indices].copy()
        self.previous_solution = self.initial_solution.copy()

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, kind, name)
        if identifier < 0:
            raise KeyError(f"Missing MuJoCo object: {name}")
        return identifier

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, Rotation]:
        return (
            data.site_xpos[self.ee_site_id].copy(),
            Rotation.from_matrix(data.xmat[self.link7_id].reshape(3, 3)),
        )

    def current_pose(self) -> tuple[np.ndarray, Rotation]:
        return self.pose(self.env.data)

    def solve(
        self, target_position: np.ndarray, target_rotation: Rotation
    ) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = self.env.data.qpos
        self.data.qvel[:] = 0.0

        def residual(joints: np.ndarray) -> np.ndarray:
            self.data.qpos[self.qpos_indices] = joints
            mujoco.mj_forward(self.model, self.data)
            position, rotation = self.pose(self.data)
            return np.concatenate(
                [
                    target_position - position,
                    (target_rotation * rotation.inv()).as_rotvec(),
                ]
            )

        current = self.env.data.qpos[self.qpos_indices].copy()
        starts = [self.previous_solution, current, self.initial_solution]
        unique_starts: list[np.ndarray] = []
        for start in starts:
            if not any(np.array_equal(start, saved) for saved in unique_starts):
                unique_starts.append(start)

        candidates: list[tuple[Any, np.ndarray]] = []
        for start in unique_starts:
            result = least_squares(
                residual,
                start,
                bounds=(self.ranges[:, 0], self.ranges[:, 1]),
                method="trf",
                max_nfev=self.max_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
            )
            error = residual(result.x)
            candidates.append((result, error))
            accurate = (
                np.linalg.norm(error[:3]) <= self.accept_position_error
                and np.linalg.norm(error[3:]) <= self.accept_rotation_error
            )
            # The previous solution is the first start. Accept it immediately
            # when it reaches the requested pose, even if SciPy only stopped
            # because max_nfev was exhausted at sub-millimetre residual.
            if accurate and len(candidates) == 1:
                self.previous_solution = result.x.copy()
                return result.x, error

        accurate_candidates = [
            item
            for item in candidates
            if np.linalg.norm(item[1][:3]) <= self.accept_position_error
            and np.linalg.norm(item[1][3:]) <= self.accept_rotation_error
        ]
        if accurate_candidates:
            result, error = min(
                accurate_candidates,
                key=lambda item: np.linalg.norm(item[0].x - self.previous_solution),
            )
            self.previous_solution = result.x.copy()
            return result.x, error

        successful_candidates = [item for item in candidates if item[0].success]
        if successful_candidates:
            result, error = min(successful_candidates, key=lambda item: np.linalg.norm(item[1]))
            self.previous_solution = result.x.copy()
            return result.x, error

        best_result, best_error = min(candidates, key=lambda item: np.linalg.norm(item[1]))
        raise RuntimeError(
            "Panda move_abs IK failed after Farm-style recovery: "
            f"{best_result.message}; position_error={np.linalg.norm(best_error[:3]):.6f} m, "
            f"rotation_error={np.linalg.norm(best_error[3:]):.6f} rad"
        )


class InferenceVisualizer:
    def __init__(self, config: dict[str, Any], encoder_mode: str) -> None:
        self.config = config
        self.encoder_mode = encoder_mode
        self.enabled = bool(config["enabled"])
        self.display = bool(config["display_window"])
        self.save = bool(config["save_frames"])
        self.scale = int(config["image_scale"])
        self.window_name = str(config["window_name"])
        self.quit_key = str(config["quit_key"])
        self.save_dir = ROOT / str(config["save_directory"])
        self.window_created = False
        if self.save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def green_mask(rgb: np.ndarray) -> np.ndarray:
        image = rgb.astype(np.float32) / 255.0
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        maximum = image.max(axis=2)
        minimum = image.min(axis=2)
        delta = maximum - minimum + 1e-6
        hue = np.zeros_like(maximum)
        red_max = maximum == red
        green_max = maximum == green
        blue_max = maximum == blue
        hue[red_max] = np.mod(
            (green[red_max] - blue[red_max]) / delta[red_max], 6
        )
        hue[green_max] = (
            (blue[green_max] - red[green_max]) / delta[green_max] + 2
        )
        hue[blue_max] = (
            (red[blue_max] - green[blue_max]) / delta[blue_max] + 4
        )
        hue /= 6.0
        hue[maximum - minimum < 1e-5] = 0.0
        saturation = delta / (maximum + 1e-6)
        saturation[maximum == 0] = 0.0
        return (
            (hue >= 0.22)
            & (hue <= 0.44)
            & (saturation >= 0.35)
            & (maximum >= 0.30)
        )

    @staticmethod
    def centroid(mask: np.ndarray) -> tuple[float, float, float, float, int]:
        height, width = mask.shape
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        mass = float(mask.sum()) + 1e-6
        y = float((mask * grid_y).sum() / mass)
        x = float((mask * grid_x).sum() / mass)
        normalized_x = x / (width - 1) * 2 - 1
        normalized_y = y / (height - 1) * 2 - 1
        return x, y, normalized_x, normalized_y, int(mask.sum())

    @staticmethod
    def write_lines(
        image: np.ndarray,
        lines: list[str],
        x: int,
        y: int,
        color: tuple[int, int, int] = (230, 230, 230),
    ) -> None:
        import cv2

        for line in lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 22

    def render(
        self,
        step: int,
        rgb_batch: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        current_position: np.ndarray,
        goal_position: np.ndarray,
        ik_residual: np.ndarray,
        inference_ms: float,
    ) -> bool:
        if not self.enabled:
            return False
        import cv2

        rgb = rgb_batch[0]
        size = 96 * self.scale
        input_panel = cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        vision_panel = input_panel.copy()
        vision_lines: list[str]
        if (
            self.encoder_mode == "green_center"
            and self.config["show_green_mask_for_green_center"]
        ):
            mask = self.green_mask(rgb)
            x, y, nx, ny, pixels = self.centroid(mask)
            mask_large = cv2.resize(
                mask.astype(np.uint8) * 255,
                (size, size),
                interpolation=cv2.INTER_NEAREST,
            )
            vision_panel = cv2.cvtColor(mask_large, cv2.COLOR_GRAY2BGR)
            point = (round(x * self.scale), round(y * self.scale))
            cv2.drawMarker(
                vision_panel,
                point,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )
            vision_lines = [
                f"Green pixels: {pixels}",
                f"Centroid px: ({x:.2f}, {y:.2f})",
                f"Keypoint: ({nx:+.4f}, {ny:+.4f}) x 32",
            ]
        else:
            vision_lines = ["ResNet18 learned feature encoder", "RGB input shown (no model hooks)"]

        canvas = np.full((size + 320, size * 2, 3), 24, dtype=np.uint8)
        canvas[:size, :size] = input_panel
        canvas[:size, size:] = vision_panel
        self.write_lines(canvas, ["MODEL RGB INPUT"], 10, 22, (80, 255, 80))
        self.write_lines(canvas, [f"VISION: {self.encoder_mode}"], size + 10, 22, (80, 255, 255))
        self.write_lines(canvas, vision_lines, size + 10, 48, (80, 255, 255))

        def values(items: np.ndarray) -> str:
            return "[" + " ".join(f"{value:+.4f}" for value in items) + "]"

        state_lines = [
            f"step={step} inference={inference_ms:.1f} ms",
            f"state width: {state[0]:.4f}",
            "state xyz: " + values(state[1:4]),
            "state rot6a: " + values(state[4:7]),
            "state rot6b: " + values(state[7:10]),
            f"action width: {action[0]:.4f}",
            "action xyz: " + values(action[1:4]),
            "action rot6a: " + values(action[4:7]),
            "action rot6b: " + values(action[7:10]),
        ]
        pose_lines = [
            "current EE xyz: " + values(current_position),
            "model goal xyz: " + values(goal_position),
            "IK xyz residual: " + values(ik_residual[:3]),
            "IK rot residual: " + values(ik_residual[3:]),
            f"IK residual norm: {np.linalg.norm(ik_residual):.6f}",
            f"Press {self.quit_key.upper()} to stop",
        ]
        self.write_lines(canvas, state_lines, 10, size + 28)
        self.write_lines(canvas, pose_lines, size, size + 28)

        if self.save:
            cv2.imwrite(str(self.save_dir / f"step_{step:06d}.png"), canvas)
        if self.display:
            cv2.imshow(self.window_name, canvas)
            self.window_created = True
            return cv2.waitKey(1) & 0xFF == ord(self.quit_key)
        return False

    def close(self) -> None:
        if self.window_created:
            import cv2

            cv2.destroyWindow(self.window_name)


class DiffusionMujocoRunner:
    def __init__(self, config: dict[str, Any], load_policy: bool = True) -> None:
        self.config = config
        runtime = config["runtime"]
        camera = config["camera"]
        third = camera["third_person"]
        gripper = config["gripper"]

        camera_config = CameraConfig(
            width=int(camera["width"]),
            height=int(camera["height"]),
            fovy=float(camera["fovy_degrees"]),
            pos=tuple(vector(camera["xyz"], 3, "camera.xyz")),
            euler=tuple(vector(camera["rpy"], 3, "camera.rpy")),
        )
        third_config = ThirdPersonCameraConfig(
            lookat=tuple(vector(third["lookat"], 3, "camera.third_person.lookat")),
            azimuth=float(third["azimuth_degrees"]),
            elevation=float(third["elevation_degrees"]),
            distance=float(third["distance"]),
        )
        self.env = FrankaUmiCameraEnv(
            camera=camera_config,
            third_person_camera=third_config,
            export_xml=bool(runtime["export_merged_xml"]),
            gripper_opening=float(gripper["initial_joint"]),
        )
        self.apply_scene(config["scene"])

        self.invert_gripper = bool(gripper["invert_joint"])
        self.gripper_max = float(gripper["joint_max"])
        self.width_scale = float(gripper["width_scale"])
        self.width_offset = float(gripper["width_offset"])
        if self.width_scale == 0:
            raise ValueError("gripper.width_scale must not be zero")

        # Optional grasp-assist loop.  It observes the simulated finger opening
        # directly; it does not change the policy image, state, or arm command.
        closure_loop = gripper.get("closure_loop", {})
        self.gripper_closure_enabled = bool(closure_loop.get("enabled", False))
        self.gripper_closure_trigger = float(
            closure_loop.get("trigger_opening_metres", 0.025)
        )
        self.gripper_closed_opening = float(
            closure_loop.get("closed_opening_metres", 0.0)
        )
        if not (
            0.0
            <= self.gripper_closed_opening
            <= self.gripper_closure_trigger
            <= self.gripper_max
        ):
            raise ValueError(
                "gripper.closure_loop openings must satisfy "
                "0 <= closed_opening_metres <= trigger_opening_metres <= joint_max"
            )
        self.gripper_closure_triggered = False

        self.control_hz = float(runtime["control_hz"])
        self.arm_command_ramp_fraction = float(runtime["arm_command_ramp_fraction"])
        if not 0 <= self.arm_command_ramp_fraction <= 1:
            raise ValueError("runtime.arm_command_ramp_fraction must be in [0, 1]")
        self.substeps = max(
            1, round((1 / self.control_hz) / self.env.model.opt.timestep)
        )
        self.ik = PandaMoveAbsIK(self.env, config["ik"])
        self.arm_command = self.env.data.qpos[self.ik.qpos_indices].copy()
        self.policy = TestDemeterPolicy(config["model"]) if load_policy else None
        encoder_mode = self.policy.encoder_mode if self.policy else config["model"]["mode"]
        self.visualizer = InferenceVisualizer(config["visualization"], encoder_mode)

    def apply_scene(self, scene: dict[str, Any]) -> None:
        support_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "support_box_body"
        )
        self.env.model.body_pos[support_id] = vector(
            scene["support_box"]["xyz"], 3, "scene.support_box.xyz"
        )
        self.env.model.body_quat[support_id] = euler_xyz_to_quat(
            vector(scene["support_box"]["rpy"], 3, "scene.support_box.rpy")
        )

        for name in ("pot", "plant"):
            joint_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint"
            )
            address = int(self.env.model.jnt_qposadr[joint_id])
            self.env.data.qpos[address : address + 3] = vector(
                scene[name]["xyz"], 3, f"scene.{name}.xyz"
            )
            self.env.data.qpos[address + 3 : address + 7] = euler_xyz_to_quat(
                vector(scene[name]["rpy"], 3, f"scene.{name}.rpy")
            )

        mount_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "panda/umi_gripper/"
        )
        self.env.model.body_pos[mount_id] = vector(
            scene["gripper_mount"]["xyz"], 3, "scene.gripper_mount.xyz"
        )
        self.env.model.body_quat[mount_id] = quat_multiply(
            euler_xyz_to_quat(GRIPPER_MOUNT_EULER),
            euler_xyz_to_quat(
                vector(
                    scene["gripper_mount"]["rpy"],
                    3,
                    "scene.gripper_mount.rpy",
                )
            ),
        )
        mujoco.mj_forward(self.env.model, self.env.data)

    def gripper_opening(self) -> float:
        """Return the current physical finger opening in metres."""
        joint = np.asarray(
            self.env.physics.named.data.qpos[FINGER_JOINT_NAMES[0]]
        ).item()
        return self.gripper_max - joint if self.invert_gripper else joint

    def gripper_width(self) -> float:
        """Convert the physical opening to the calibrated policy input unit."""
        return self.width_scale * self.gripper_opening() + self.width_offset

    def gripper_joint_target_from_opening(self, opening: float) -> float:
        """Convert an opening in metres to the MuJoCo finger actuator target."""
        return self.gripper_max - opening if self.invert_gripper else opening

    def gripper_target(self, model_width: float) -> float:
        opening = (model_width - self.width_offset) / self.width_scale
        return self.gripper_joint_target_from_opening(opening)

    def apply_gripper_closure_loop(self, model_target: float) -> float:
        """Force full closure once the measured opening reaches the threshold."""
        if not self.gripper_closure_enabled:
            return model_target
        opening = self.gripper_opening()
        if opening > self.gripper_closure_trigger:
            return model_target
        if not self.gripper_closure_triggered:
            print(
                "Gripper closure loop triggered: "
                f"opening={opening:.4f} m <= "
                f"{self.gripper_closure_trigger:.4f} m"
            )
            self.gripper_closure_triggered = True
        return self.gripper_joint_target_from_opening(self.gripper_closed_opening)

    def step_control_cycle(
        self, arm_target: np.ndarray, gripper_target: float
    ) -> None:
        """Ramp early within one cycle, then hold the exact model target.

        This preserves finalv1's observation-to-action timing: the complete IK
        target is applied before the next policy observation. The short ramp
        only prevents an instantaneous high-gain actuator target jump.
        """
        start = self.arm_command.copy()
        ramp_steps = max(1, round(self.substeps * self.arm_command_ramp_fraction))
        for substep in range(self.substeps):
            alpha = min((substep + 1) / ramp_steps, 1.0)
            command = start + alpha * (arm_target - start)
            closed_loop_target = self.apply_gripper_closure_loop(gripper_target)
            self.env.set_control(arm=command, gripper=closed_loop_target)
            self.env.step_simulation(1)
        self.arm_command = arm_target.copy()

    def observation(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        import cv2

        position, rotation = self.ik.current_pose()
        raw_rgb = self.env.get_camera_image(
            height=self.env.camera.height, width=self.env.camera.width
        )
        rgb_batch = np.array(
            [cv2.resize(raw_rgb, (96, 96), interpolation=cv2.INTER_AREA)],
            dtype=np.uint8,
        )
        return self.gripper_width(), rgb_batch, position, rotation.as_quat()

    def dry_run(self) -> None:
        width, image, position, quaternion = self.observation()
        state = np.concatenate(
            [[width], position, rotation_to_feature(Rotation.from_quat(quaternion))]
        )
        print("Dry run OK")
        print("state:", state)
        print("image shape/dtype/range:", image.shape, image.dtype, (image.min(), image.max()))
        print("control_hz/substeps:", self.control_hz, self.substeps)
        print(
            "gripper closure loop:",
            self.gripper_closure_enabled,
            "trigger opening (m):",
            self.gripper_closure_trigger,
            "closed opening (m):",
            self.gripper_closed_opening,
        )

    def run(self) -> None:
        if self.policy is None:
            raise RuntimeError("Policy was not loaded")
        runtime = self.config["runtime"]
        viewer = self.env.launch_passive_viewer() if runtime["viewer"] else None
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = self.env.third_person_camera.lookat
            viewer.cam.azimuth = self.env.third_person_camera.azimuth
            viewer.cam.elevation = self.env.third_person_camera.elevation
            viewer.cam.distance = self.env.third_person_camera.distance
            viewer.opt.geomgroup[3] = 0
            # Show only the main plant collision capsule in the third-person
            # viewer; the model camera still excludes debug group 5.
            viewer.opt.geomgroup[5] = 1

        try:
            for step in range(int(runtime["steps"])):
                if viewer is not None and not viewer.is_running():
                    break
                cycle_start = time.perf_counter()
                width, image, current_position, current_quaternion = self.observation()
                inference_start = time.perf_counter()
                state, action = self.policy.select_action(
                    width, image, current_position, current_quaternion
                )
                inference_ms = (time.perf_counter() - inference_start) * 1000

                goal_position = action[1:4].astype(float)
                goal_rotation = feature_to_rotation(action[4:10])
                arm_target, ik_residual = self.ik.solve(goal_position, goal_rotation)
                gripper_target = self.gripper_target(float(action[0]))

                stop = self.visualizer.render(
                    step,
                    image,
                    state,
                    action,
                    current_position,
                    goal_position,
                    ik_residual,
                    inference_ms,
                )
                self.step_control_cycle(arm_target, gripper_target)
                if viewer is not None:
                    viewer.sync()

                if step % int(runtime["log_every_steps"]) == 0:
                    print(
                        f"step={step} inference_ms={inference_ms:.1f} "
                        f"goal_width={action[0]:.3f} goal_xyz={goal_position} "
                        f"ik_error={np.linalg.norm(ik_residual):.6f}"
                    )
                if stop:
                    break
                if runtime["realtime"]:
                    delay = 1 / self.control_hz - (time.perf_counter() - cycle_start)
                    if delay > 0:
                        time.sleep(delay)
        finally:
            if viewer is not None:
                viewer.close()
            self.visualizer.close()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runner = DiffusionMujocoRunner(
        read_config(args.config.resolve()), load_policy=not args.dry_run
    )
    runner.dry_run() if args.dry_run else runner.run()


if __name__ == "__main__":
    main()
