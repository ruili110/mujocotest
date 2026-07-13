import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2  # 引入 OpenCV 处理图像

# 1. 加载场景
xml_path = "mujoco_menagerie/franka_emika_panda/grasp_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. 初始化离屏渲染器 (Renderer)
# 高度和宽度可以根据你的扩散模型需求来调整，通常扩散模型使用 224x224 或 256x256
renderer = mujoco.Renderer(model, height=480, width=640)

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()

        # =======================================================
        # 这里放置你的控制逻辑 (比如后续接入摇杆或者扩散模型输出的 Action)
        # 测试：让机械臂随机轻微抖动，方便我们观察相机画面的变化
        data.ctrl[0] = np.sin(time.time()) * 0.1
        data.ctrl[1] = 0.5
        # =======================================================
        
        # 推进物理引擎
        mujoco.mj_step(model, data)
        viewer.sync()

        # =======================================================
        # 核心逻辑：获取腕部相机的图像数据
        # =======================================================
        # 更新渲染器场景，指定我们刚才在 XML 里命名的相机 "wrist_cam"
        renderer.update_scene(data, camera="wrist_cam")
        
        # 提取像素矩阵 (RGB 格式, 形状为 Height x Width x 3 的 numpy array)
        pixels = renderer.render()
        
        # 这一步 pixels 变量就是你未来喂给扩散模型的输入数据！
        # print(pixels.shape) # 输出: (480, 640, 3)

        # 为了方便调试，我们用 OpenCV 实时把相机画面弹出来
        # 注意：OpenCV 默认使用 BGR 格式显示图像，所以我们要把 MuJoCo 的 RGB 转成 BGR
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("Wrist Camera View (For Diffusion Model)", bgr_pixels)
        cv2.waitKey(1)  # 保持图像窗口刷新
        # =======================================================

        # 保持时间同步
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# 退出循环后关闭图像窗口
cv2.destroyAllWindows()