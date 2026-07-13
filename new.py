# 运行前请确保安装了依赖: pip install mujoco dm_control
from dm_control import mjcf
import mujoco
import mujoco.viewer

def build_and_run_scene():
    print("正在加载模型...")
    # 1. 载入三个独立的模型文件
    # 注意：确保这三个文件与 Python 脚本在同一目录下，或者提供绝对路径
    env = mjcf.from_path("C:/Users/Lirui/Desktop/franka_mujoco/mujoco_menagerie/franka_emika_panda/plant_env.xml")
    arm = mjcf.from_path("C:/Users/Lirui/Desktop/franka_mujoco/mujoco_menagerie/franka_emika_panda/panda_no_gripper.xml")
    gripper = mjcf.from_path("C:/Users/Lirui/Desktop/franka_mujoco/mujoco_menagerie/umi_gripper/umi_gripper.xml")
    for j in gripper.find_all('joint'):
        if j.type == 'free':
            j.remove()
            print("✓ 已移除 UMI 爪子的独立自由关节 (freejoint)")



    # 2. 将 UMI 爪子嫁接到 Franka 的法兰盘上
    # 找到 Franka 的末端法兰连杆 link7 (根据你的 XML 实际情况)
    link7 = arm.find('body', 'link7')
    
    # 在 link7 上创建一个叫 'gripper_mount' 的挂载点 (Site)
    # 【核心修改】直接照搬原装 hand 的 pos 和 quat 参数，保证新爪子位置和朝向绝对正确！
    mount_site = link7.add('site', name='gripper_mount', pos=[0, 0, 0.107], euler=[0, 0, 2.356])
    
    # 核心魔法：使用 attach 方法，自动处理所有的电机、材质命名冲突！
    mount_site.attach(gripper)
    print("✓ 机械臂与爪子拼接成功！")

    # 3. 将组装好的“机械臂+爪子”放进植物环境里
    # 获取场景的全局坐标系 (worldbody)
    worldbody = env.worldbody
    
    # 在地面的原点创建一个给机械臂使用的基座挂载点
    arm_base_site = worldbody.add('site', name='arm_base', pos=[0, 0, 0])
    arm_base_site.attach(arm)
    print("✓ 机械臂成功放入植物场景！")

    # 4. (可选但推荐) 导出拼接后的完整 XML 以供后续直接使用或检查
    with open("merged_full_scene.xml", "w") as f:
        f.write(env.to_xml_string())
    print("✓ 完整模型已导出至 merged_full_scene.xml")

    # 5. 生成物理对象并启动可视化器
    print("正在启动 MuJoCo 仿真...")
    physics = mjcf.Physics.from_mjcf_model(env)
    
    # 启动交互窗口
    mujoco.viewer.launch(physics.model.ptr, physics.data.ptr)

if __name__ == "__main__":
    build_and_run_scene()