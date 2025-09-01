# 开源重定向项目-PBHC

**项目地址：**https://github.com/TeleHuman/PBHC

**项目优势：**在不考虑手部(L/R_Wrist)运动(joint_fixed)，或认为手部为刚性连接(运动很小)时，具有良好的效果

**项目缺点：**由于本身为基于24自由度的G1所编写，在重定向过程几乎不考虑方向，只考虑位置，因此在改为29自由度后，手部转动(left/right_pitch/roll/yaw_joint)，脚踝转动(left/right_ankle_roll/yaw_joint)不明显或不动，**暂时无有效解**。

## 部署

```bash
#从项目仓库中拉取代码到本地
git clone https://github.com/TeleHuman/PBHC.git
cd PBHC
#安装项目
##安装大环境：./INSTALL.md

##安装重定向主要程序
cd smpl_retarget/poselib
pip install -e .
cd ..
pip install chumpy
git clone https://github.com/ZhengyiLuo/SMPLSim.git
pip install -e ./SMPLSim
#将smpl_mode文件夹复制到目录PBHC/smpl_retarget/下
```

## 重定向

PBHC提供基于mink与PHC两种重定向方式，以下内容均基于mink书写(mink_retarget)。

在进行重定向之前，我们需要针对本实验室29自由度的g1机器人进行修改。

**创建文件dof_axis_full29.npy	路径PBHC/description/robots/g1**

```bash
#运行下列代码，或在根目录下保存为*.py文件并运行
python -c "
import numpy as np
arr = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
np.save('description/robots/g1/dof_axis_full29.npy', arr)
"
#创建完整29关节的转轴数组
```

**文件convert_fit_motion.py	路径PBHC/smpl_retarget/mink_retarget**

```bash
#修改1，行51
#原文：dof_new = np.concatenate((dof[:, :19], dof[:, 22:26]), axis=1) 改为
dof_new = dof.copy()
#原因：dof_new截取了机器人关节数据的部分内容，删除了left_wrist_roll/pitch/yaw_joint(dof[:,20:22])与right_wirst_roll/pitch/yaw_joint(dof[:,27:29])的数据，若需完成29自由度，需保留全部内容

#修改2，行54
#原文：dof_axis = np.load('../description/robots/g1/dof_axis.npy', allow_pickle=True)
dof_axis = np.load('../description/robots/g1/dof_axis_full29.npy', allow_pickle=True)
#原因：原始dof_axis.npy文件只包含23自由度的关节转轴，需替换为29自由度的关节转轴

#修改3，行298
#原文：pose_aa = np.concatenate(...)
pose_aa = motion_data["pose_aa"][:,:72]
#原因：pose_aa删除了原始smpl模型的L/R_Wrist的运动数据并补零，使得重定向结果手腕呈刚性连接，补全以获得腕部数据


```

**文件mink_retarget.py		路径PBHC/smpl_retarget/mink_retarget/retargeting**

```bash
#修改1，行466
#原文：orientation_cost=orientation_base_cost * retarget_info["weight"]
orientation_cost=orientation_base_cost * retarget_info["weight"] if "Wrist" not in joint_name else 1 * retarget_info["weight"],
#原因：原始方法对于方向的权重极低，为万分之一，若需使手腕遵循原始数据运动，则需调整至合适大小

#修改2，行574
#在该行后新建一行，补充如下代码
if "Wrist" in joint_name:
	correction_quat = sRot.from_euler('z', 90 if joint_name=="L_Wrist" else -90, degrees=True).as_quat()
    target_rot = sRot.from_quat(target_rot) * sRot.from_quat(correction_quat)
    target_rot = target_rot.as_quat()
#原因：原始方法对于腕部left/right_wrist_yaw_joint的默认方向定义错误，存在+-90度的差异，在此修正
```

**创建运动数据文件夹<motion_data>**，完成后应为

```bash
motion_data/
├── your_folder_name_1/
|    └── your_data.npz
|    └── ...
├── your_folder_name_2/
|    └── your_data.npz
|    └── ...
└──...
```

完成上述修正后，可以使用下列代码重定向

```bash
python mink_retarget/convert_fit_motion.py <PATH_TO_MOTION_DATA>

#若需二次重定向并覆盖原始内容，可添加 --force-remake 或删除 motion_data/<your_folder_name>-g1_retargeted_npy/目录下的*.npy文件

#若出现报错：ImportError: cannot import name 'bool' from 'numpy'且文件为：/home/user/anaconda3/envs/pbhc/lib/python3.8/site-packages/chumpy/__init__.py则执行以下命令，降低numpy版本
pip uninstall numpy
pip install "numpy<1.23"
```

输出结果位于PBHC/smpl_retarget/retargeted_motion_data/mink目录下，为\*.pkl文件。



## 可视化重定向

在PBHC/robot_motion_process/目录下，提供对于重定向输出文件\*.pkl的可视化，针对29自由度的文件，进行以下修改

**在目录PBHC/description/robot/g1下创建phc_g1_29dof.yaml**

```bash
#复制下列内容至phc_g1_29dof.yaml
humanoid_type: g1_29dof_anneal
bias_offset: False
has_self_collision: True
has_mesh: False
has_jt_limit: False
has_dof_subset: True
has_upright_start: True
has_smpl_pd_offset: False
remove_toe: False
motion_sym_loss: False
sym_loss_coef: 1
big_ankle: True

has_shape_obs: false
has_shape_obs_disc: false
has_shape_variation: False

masterfoot: False
freeze_toe: false
freeze_hand: false   
box_body: True
real_weight: True
real_weight_porpotion_capsules: True
real_weight_porpotion_boxes: True

body_names: [ 'pelvis',
              'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
              'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
              'waist_yaw_link', 'waist_roll_link', 'torso_link',
              'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link','left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
              'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link' ,'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']

dof_names:
  - 'left_hip_pitch_link'
  - 'left_hip_roll_link'
  - 'left_hip_yaw_link'
  - 'left_knee_link'        
  - 'left_ankle_pitch_link'
  - 'left_ankle_roll_link'
  - 'right_hip_pitch_link'
  - 'right_hip_roll_link'
  - 'right_hip_yaw_link'
  - 'right_knee_link'
  - 'right_ankle_pitch_link'
  - 'right_ankle_roll_link'
  - 'waist_yaw_link'
  - 'waist_roll_link'
  - 'torso_link'
  - 'left_shoulder_pitch_link'
  - 'left_shoulder_roll_link'
  - 'left_shoulder_yaw_link'
  - 'left_elbow_link'
  - 'left_wrist_roll_link'
  - 'left_wrist_pitch_link'
  - 'left_wrist_yaw_link'
  - 'right_shoulder_pitch_link'
  - 'right_shoulder_roll_link'
  - 'right_shoulder_yaw_link'
  - 'right_elbow_link'
  - 'right_wrist_roll_link'
  - 'right_wrist_pitch_link'
  - 'right_wrist_yaw_link'

right_foot_name: 'r_foot_roll'
left_foot_name:  'l_foot_roll'

limb_weight_group:
  - [ 'left_hip_pitch_link',  'left_hip_roll_link',  'left_hip_yaw_link',
      'left_knee_link',       'left_ankle_pitch_link','left_ankle_roll_link' ]
  - [ 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
      'right_knee_link',      'right_ankle_pitch_link','right_ankle_roll_link' ]
  - [ 'pelvis','waist_yaw_link','waist_roll_link','torso_link' ]
  - [ 'left_shoulder_pitch_link','left_shoulder_roll_link','left_shoulder_yaw_link','left_elbow_link',
      'left_wrist_roll_link','left_wrist_pitch_link','left_wrist_yaw_link' ]
  - [ 'right_shoulder_pitch_link','right_shoulder_roll_link','right_shoulder_yaw_link','right_elbow_link',
      'right_wrist_roll_link','right_wrist_pitch_link','right_wrist_yaw_link' ]

asset:
  assetRoot: "./"
  assetFileName: "description/robots/g1/g1_29dof_rev_1_0.xml"   
  urdfFileName:  "description/robots/g1/g1_29dof_rev_1_0.xml"

extend_config:
  - joint_name: "left_hand_link"
    parent_name: "left_wrist_yaw_link"
    pos: [0.08, 0.0, 0.0]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "right_hand_link"
    parent_name: "right_wrist_yaw_link"
    pos: [0.08, 0.0, 0.0]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "head_link"
    parent_name: "torso_link"
    pos: [0.0, 0.0, 0.42]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "left_toe_link"
    parent_name: "left_ankle_roll_link"
    pos: [0.08, 0, -0.06]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "right_toe_link"
    parent_name: "right_ankle_roll_link"
    pos: [0.08, 0, -0.06]
    rot: [1.0, 0.0, 0.0, 0.0]

base_link: "torso_link"

joint_matches:
  - [ "pelvis", "Pelvis" ]
  - [ "left_hip_pitch_link", "L_Hip" ]
  - [ "left_knee_link", "L_Knee" ]
  - [ "left_ankle_roll_link", "L_Ankle" ]
  - [ "right_hip_pitch_link", "R_Hip" ]
  - [ "right_knee_link", "R_Knee" ]
  - [ "right_ankle_roll_link", "R_Ankle" ]
  - [ "left_shoulder_roll_link", "L_Shoulder" ]
  - [ "left_elbow_link", "L_Elbow" ]
  - [ "left_wrist_yaw_link", "L_Wrist" ]
  - [ "left_hand_link", "L_Hand" ]
  - [ "right_shoulder_roll_link", "R_Shoulder" ]
  - [ "right_elbow_link", "R_Elbow" ]
  - [ "right_wrist_yaw_link", "R_Wrist" ]
  - [ "right_hand_link", "R_Hand" ]
  - [ "head_link", "Head" ]
  - [ "left_toe_link", "L_Toe" ]
  - [ "right_toe_link", "R_Toe" ]

smpl_pose_modifier:
  Pelvis:      "[np.pi/2, 0, np.pi/2]"
  L_Shoulder:  "[0, 0, -np.pi/2]"
  R_Shoulder:  "[0, 0,  np.pi/2]"
  L_Elbow:     "[0, -np.pi/2, 0]"
  R_Elbow:     "[0,  np.pi/2, 0]"
  L_Wrist:     "[0, 0, 0]"
  R_Wrist:     "[0, 0, 0]"
```

**文件g1_29dof_rev_1_0.xml	路径description/robots/g1**

```bash
#解除setup scene的全部注释
  setup scene
  <statistic center="1.0 0.7 1.0" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
#原因：在可视化中保留地板等环境的设置

```

**文件vis_q_mj.py	路径PBHC/robot_motion_process**

```bash
#修改1，行113
#原文：humanoid_xml = "./description/robots/g1/g1_23dof_lock_wrist.xml"
humanoid_xml = "./description/robots/g1/g1_29dof_rev_1_0.xml"
#原因：导入全部关节文件

#修改2，行126
#原文：cfg_robot = OmegaConf.load("description/robots/g1/phc_g1_23dof.yaml")
cfg_robot = OmegaConf.load("description/robots/g1/phc_g1_29dof.yaml")
#原因：令可视化关节获取正确的关节名称，初始位置和smpl修正

#修改3，行181
#原文：for i in range(23):
for i in range(29):
#原因：展示所有关节
```

***可选修改**

若不想看到仿真中的关节映射点（红球），可以注释如下代码

**文件vis_q_mj.py	路径PBHC/robot_motion_process**

```bash
#行175
if vis_smpl:
    joint_gt = motion_data[curr_motion_key]['smpl_joints']
    if not np.all(joint_gt[curr_time] == 0):
        for i in range(joint_gt.shape[1]):
            viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
else:
	for i in range(29):
        viewer.user_scn.geoms[i+1].pos = joint_gt[curr_time, i+1]
```

即可关闭显示

完成上述操作后，可以使用如下代码可视化

```bash
python robotpython robot_motion_process/vis_q_mj.py +motion_file=<PATH_TO_YOUR_FILE>.pkl
```



## 重定向的参数与调整

### 全局参数

文件conver_fit_motion.py完成原始数据读取与转化，重定向结果的保存，文件mink_retarget.py完成重定向主要工作，包括逆运动学和碰撞检测。

```bash
#mink_retarget.py，行69
_G1_KEYPOINT_TO_JOINT = {
    "Pelvis": {"name": "pelvis", "weight": 5.0},
    "Head": {"name": "head", "weight": 5.0},
    # Legs.
    "L_Hip": {"name": "left_hip_yaw_link", "weight": 1.0},
    "R_Hip": {"name": "right_hip_yaw_link", "weight": 1.0},
    "L_Knee": {"name": "left_knee_link", "weight": 1.0},
    "R_Knee": {"name": "right_knee_link", "weight": 1.0},
    "L_Ankle": {"name": "left_ankle_roll_link", "weight": 1.0},
    "R_Ankle": {"name": "right_ankle_roll_link", "weight": 1.0},
    # Arms.
    "L_Elbow": {"name": "left_elbow_link", "weight": 1.0},
    "R_Elbow": {"name": "right_elbow_link", "weight": 1.0},
    "L_Wrist": {"name": "left_wrist_yaw_link", "weight": 1.0},
    "R_Wrist": {"name": "right_wrist_yaw_link", "weight": 1.0},
    "L_Shoulder": {"name": "left_shoulder_pitch_link", "weight": 3.0},
    "R_Shoulder": {"name": "right_shoulder_pitch_link", "weight": 3.0},

    # toe
    "L_Toe": {"name": "left_toe_link", "weight": 1.0},
    "R_Toe": {"name": "right_toe_link", "weight": 1.0},
    # torso
    # "Torso": {"name": "torso_link", "weight": 3.0},

    # Hands
    # "L_Hand": {"name": "left_rubber_hand_2", "weight": 3.0},
    # "R_Hand": {"name": "right_rubber_hand_2", "weight": 3.0},
}
```

**该数组存放了g1重定向的全局权重信息，格式为"\<SMPL关节名称>": {"name": "\<G1关节名称>", "weight": \<关节权重>}。**

### 任务设置

```bash
#mink_retarget.py，行457
for joint_name, retarget_info in _KEYPOINT_TO_JOINT_MAP[robot_type].items():
    if robot_type == "h1":
    orientation_base_cost = 0
    else:
        orientation_base_cost = 0.0001
    task = mink.FrameTask(
        frame_name=retarget_info["name"],
        frame_type="body",
        position_cost=10.0 * retarget_info["weight"],
        orientation_cost=orientation_base_cost * retarget_info["weight"],
        lm_damping=1.0,
    )
    frame_tasks[retarget_info["name"]] = task
tasks.extend(frame_tasks.values())
```

该段代码向求解器中添加每个SMPL关节的运动求解，joint_name遍历每一个在上述_G1_KEYPOINT_TO_JOINT中存放的SMPL关节名称，并计算各自的位置权重

```bash
position_cost = 10.0 * retarget_info["weight"]
综合位置权重 = 基础位置权重常数 * 特定位置全局权重

orientation_cost = orientation_base_cost * retarget_info["weight"]
综合方向权重 = 基础方向权重常数 * 特定位置全局权重
```

**由于原始重定向中关于方向权重的设定很小，故在29关节重定向实践中，若使用该方法，则需要调大至0.01-1中的合适数值或只针对特定关节调整权重。**

### 碰撞对检测

本项目的碰撞检测由mink支持，但是未在项目中做出具体实现，故仅供参考

```bash
#mink_retarget.py，行534
        # collision_pairs = [
        #     (["right_toe_link"], ["ground"]),
        #     (["left_toe_link"], ["ground"]),
        #     (["right_thigh_collision"], ["left_shank_collision"])

        # ]

        # # for i in range(model.nbody):
        # #     for j in range(i+1, model.nbody):
        # #         geoms_i = get_body_geom_ids(model, i)
        # #         geoms_j = get_body_geom_ids(model, j)
        # #         if geoms_i and geoms_j:
        # #             collision_pairs.append((geoms_i, geoms_j))

        # # # print(collision_pairs)

        # collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        #     model,
        #     collision_pairs,
        #     gain=0.85,
        #     minimum_distance_from_collisions=0.005,
        #     collision_detection_distance=0.01,
        #     bound_relaxation=0.0
        # )
```

在原始mink重定向中，碰撞检测被关闭，脚部触地检测在conver_fit_motion.py中进行，若需要则在重定向代码后加上--correct即可，但实测效果不理想。
