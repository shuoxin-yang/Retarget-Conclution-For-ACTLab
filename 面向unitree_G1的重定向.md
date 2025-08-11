# 面向unitree_G1的重定向

## 源文件的数据处理（\*.npz/\*.bvh）

针对不同的源文件，下述重定向具有不同的使用方式和修改内容。

### (1)	SMPL/SMPL-X(*.npz)

SMPL-X为SMPL的拓展版本，**SMPL-X格式天然支持SMPL重定向<u>（帧率除外）</u>**，且SMPL-X相比于SMPL具有更多的关节自由度。

**原生基于SMPL-X**的重定向：关节数据(pose_body or pose_aa)来自*.npz/pose_body.npy (N,63)，全局转动(global_orient)来自\*.npz/root_orient.npy (N,3)，帧率(fps or mocap_frame_rate)来自\*.npz/mocap_frame_rate.npy

**原生基于SMPL**的重定向：关节数据(pose_body or pose_aa)来自\*.npz/poses.npy的后69维(N,3:72)，全局转动来自\*.npz/poses.npy的前三维(N,0:3)，帧率(fps or mocap_framerate)来自\*.npz/mocap_framerate.npy

将基于SMPL**映射**到SMPL-X：

```bash
smplx_data["global_orient"] = smpl_data["poses"][:,:3]#映射全局转动
smplx_data["pose_body"] = smpl_data["poses"][:,3:66]#映射主要关节
smplx_data["mocap_frame_rate"] = smpl_data["mocap_framerate"]#映射帧率
#important#
#smpl_data["poses"][:,66:72]为L_Hand/R_Hand，在本次重定向不考虑，对应smplx_data["pose_hand"]
```

### (2)	动作捕捉(\*.bvh)

PBHC不支持.bvh的重定向，故以下分析基于GMR(GeneralMotionRetargeting)。

\*.bvh为文本文档格式，可直接使用VSCode阅读。

```bash
HIERARCHY
ROOT Hips
{
	OFFSET * * *
	CHANNELS *
	JOINT <joint_name_1>
	{
		OFFSET *
		***
	}
	JOINT <joint_name_2>
	{
		OFFSET *
		***
	}
}
MOTION
Frames: <motion_frames_count>
Frame Time: <second_pre_frame>
-224.689499 91.882057 -431.625488 91.911438 5.797277 88.877003 -173.825528 ...<Frame_data>
```

以上为一个标准的.bvh文件在文本阅读器里面的展示，joint可以嵌套，frame_data以单一空格区分，换行为一帧完结，**行尾不包含任何出换行符(\n)以外的任何字符，包括空格。**

## 开源重定向项目

### (1) 	PBHC

<url>https://github.com/TeleHuman/PBHC</url>

**项目优势：**在不考虑手部(L/R_Wrist)运动(joint_fixed)，或认为手部为刚性连接(运动很小)时，具有良好的效果

**项目缺点：**由于本身为基于24自由度的G1所编写，在重定向过程几乎不考虑方向，只考虑位置，因此在改为29自由度后，手部转动(left/right_pitch/roll/yaw_joint)，脚踝转动(left/right_ankle_roll/yaw_joint)不明显或不动，**暂时无有效解**。

#### 部署

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

#### 重定向

PBHC提供基于mink与PHC两种重定向方式，以下内容均基于mink书写(mink_retarget)。

在进行重定向之前，我们需要针对本实验室29自由度的g1机器人进行修改。

创建文件dof_axis_full29.npy	路径PBHC/description/robots/g1

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

文件convert_fit_motion.py	路径PBHC/smpl_retarget/mink_retarget

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

文件mink_retarget.py		路径PBHC/smpl_retarget/mink_retarget/retargeting

```bash
#修改1，行466
#原文：orientation_cost=orientation_base_cost * retarget_info["weight"]
orientation_cost=orientation_base_cost * retarget_info["weight"] if "Wrist" not in joint_name else 1 * retarget_info["weight"],
#原因：原始方法对于方向的权重极低，为万分之一，若需使手腕遵循原始数据运动，则需调整至合适大小

#修改2，行547
#在该行后新建一行，补充如下代码
if "Wrist" in joint_name:
	correction_quat = sRot.from_euler('z', 90 if joint_name=="L_Wrist" else -90, degrees=True).as_quat()
    target_rot = sRot.from_quat(target_rot) * sRot.from_quat(correction_quat)
    target_rot = target_rot.as_quat()
#原因：原始方法对于腕部left/right_wrist_yaw_joint的默认方向定义错误，存在+-90度的差异，在此修正
```

创建运动数据文件夹<motion_data>，完成后应为

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



#### 可视化重定向

在PBHC/robot_motion_process/目录下，提供对于重定向输出文件\*.pkl的可视化，针对29自由度的文件，进行以下修改

在目录PBHC/description/robot/g1下创建phc_g1_29dof.yaml

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

文件g1_29dof_rev_1_0.xml	路径description/robots/g1

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

文件vis_q_mj.py	路径PBHC/robot_motion_process

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



#### 关于重定向

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

该数组存放了g1重定向的全局权重信息，格式为"\<SMPL关节名称>": {"name": "\<G1关节名称>", "weight": \<关节权重>}。

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

### (2)GMR - GeneralMotionRetargeting

<url>https://github.com/YanjieZe/GMR</url>

**项目优势：**简单易部署，兼容SMPL-X(\*.npz)与Lafan(*.bvh)的原始数据格式，**关节转动效果较好**，脚趾与地板接触良好。

**项目缺点：**暂时没有遇到

#### 部署

```bash
#从项目仓库中拉取代码到本地
git clone https://github.com/YanjieZe/GMR.git
cd GMR
#安装项目
#创建虚拟环境
conda create -n gmr python=3.10 -y
conda activate gmr
#安装主项目
pip install -e .
#注意：如果你只有.npz或.pkl的SMPL-X标准模型，请修改smplx/body_models.py下的“ext”为对应文件格式，若两者都有，可忽略
#一些可能的依赖
conda install -c conda-forge libstdcxx-ng -y
```

#### 重定向（基于SMPL/SMPL-X）

本项目天然支持29自由度的g1机器人重定向，基于SMPL-X数据集，故若需要使用SMPL数据集，需要进行如下修改，在保留SMPL-X格式的同时，兼容SMPL。

创建目录GMR/assert/body_models/smplx/

```bash
GMR/assert/body_models/smplx/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_FEMALE.pkl
└──SMPLX_MALE.pkl
#建议一并将*.npz文件一同放入，保证最大兼容性
```

完成上述内容后，已经可以正常使用SMPL-X数据集（如AMASS）进行重定向，若需要使用SMPL格式（如GVHMR产出的），需要进行一些修改

文件smpl.py	路径GMR/general_motion_retargeting/utils

```bash
#修改1，行21
#原文
    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
#修改为
    if "pose_body" not in smplx_data.keys():
        print(smplx_data["betas"].shape)
        num_frames = smplx_data["poses"].shape[0]
        smplx_output = body_model(
            betas = torch.cat([torch.tensor(smplx_data["betas"]).float().view(1, -1), torch.zeros(1, 6)], dim=1), # (16,)
            global_orient=torch.tensor(smplx_data["poses"][:,:3]).float(), # (N, 3)
            body_pose=torch.tensor(smplx_data["poses"][:,3:66]).float(), # (N, 63)
            transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )
    else:
        print(smplx_data["betas"].shape)
        num_frames = smplx_data["pose_body"].shape[0]
        smplx_output = body_model(
            betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
            global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
            body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
            transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )
#原因：关键字“pose_body”不存在于SMPL格式中，相关内容存放在“poses”下，因此检测此关键字可以区分SMPL与SMPL-X格式，在保留原始支持SMPL—X的内容下，兼容SMPL
#注意：SMPL-X格式的“betas”关键字有16维，描述各个关节的运动姿势，而SMPL下只有10维，目前采取补0的策略，也可以使用SMPL-X文件的betas.npy替换SMPL的betas.npy

#修改2，行118(未进行修改1)/136(进行修改1)
#原文：src_fps = smplx_data["mocap_frame_rate"].item()
src_fps = smplx_data["mocap_frame_rate"].item() if "mocap_frame_rate" in smplx_data.keys() else smplx_data["mocap_framerate"].item()
#原因：SMPL格式下帧率关键字与SMPL-X不一致，需要判断并正确读取

#修改3，行120(未进行修改1)/138(进行修改1)
#原文：num_frames = smplx_data["pose_body"].shape[0]
num_frames = smplx_data["pose_body"].shape[0] if "pose_body" in smplx_data.keys() else smplx_data["poses"].shape[0]
#原因：总帧数通过统计动作的数量计算得出，可以使用上述方法，也可以直接统计“poses”关键字的长度，即：num_frames = smplx_data["poses"].shape[0]
```

完成上述内容后，使用以下代码重定向

```bash
python scripts/smplx_to_robot.py --smplx_file <PATH_TO_YOUR_SMPL_FILE>.npz --robot unitree_g1 --save_path <PATH_TO_YOUR_OUTPUT_FILE>.pkl
```

更多可选参数详见GMR/README.md

#### 重定向（基于LaFan/动作捕捉）

注意：由于各家动作捕捉生成的\*.bvh格式不统一，本项目基于LaFan格式编写，如果需要，请先进行文件格式转换，以下给出符合LaFan格式的文件样例

```bash
#example.bvh
HIERARCHY
ROOT Hips
{
	OFFSET -224.689499 91.882057 -431.625488
	CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation 
	JOINT LeftUpLeg
	{
		OFFSET 0.103456 1.857840 10.548509
		CHANNELS 3 Zrotation Yrotation Xrotation
		JOINT LeftLeg
		{
			OFFSET 43.500000 -0.000042 0.000010
			CHANNELS 3 Zrotation Yrotation Xrotation
			JOINT LeftFoot
			{
				OFFSET 42.372192 0.000019 0.000000
				CHANNELS 3 Zrotation Yrotation Xrotation
				JOINT LeftToe
				{
					OFFSET 17.299999 -0.000000 0.000003
					CHANNELS 3 Zrotation Yrotation Xrotation
					End Site
					{
						OFFSET 0.000000 0.000000 0.000000
					}
				}
			}
		}
	}
	JOINT RightUpLeg
	{
		OFFSET 0.103456 1.857823 -10.548508
		CHANNELS 3 Zrotation Yrotation Xrotation
		JOINT RightLeg
		{
			OFFSET 43.500042 -0.000027 0.000004
			CHANNELS 3 Zrotation Yrotation Xrotation
			JOINT RightFoot
			{
				OFFSET 42.372261 0.000000 0.000010
				CHANNELS 3 Zrotation Yrotation Xrotation
				JOINT RightToe
				{
					OFFSET 17.299994 -0.000006 0.000017
					CHANNELS 3 Zrotation Yrotation Xrotation
					End Site
					{
						OFFSET 0.000000 0.000000 0.000000
					}
				}
			}
		}
	}
	JOINT Spine
	{
		OFFSET 6.901967 -2.603732 -0.000003
		CHANNELS 3 Zrotation Yrotation Xrotation
		JOINT Spine1
		{
			OFFSET 12.588099 0.000010 0.000008
			CHANNELS 3 Zrotation Yrotation Xrotation
			JOINT Spine2
			{
				OFFSET 12.343201 -0.000018 -0.000005
				CHANNELS 3 Zrotation Yrotation Xrotation
				JOINT Neck
				{
					OFFSET 25.832890 0.000023 0.000007
					CHANNELS 3 Zrotation Yrotation Xrotation
					JOINT Head
					{
						OFFSET 11.766609 -0.000008 -0.000006
						CHANNELS 3 Zrotation Yrotation Xrotation
						End Site
						{
							OFFSET 0.000000 0.000000 0.000000
						}
					}
				}
				JOINT LeftShoulder
				{
					OFFSET 19.745909 -1.480347 6.000101
					CHANNELS 3 Zrotation Yrotation Xrotation
					JOINT LeftArm
					{
						OFFSET 11.284133 0.000018 -0.000020
						CHANNELS 3 Zrotation Yrotation Xrotation
						JOINT LeftForeArm
						{
							OFFSET 33.000050 -0.000013 0.000019
							CHANNELS 3 Zrotation Yrotation Xrotation
							JOINT LeftHand
							{
								OFFSET 25.200005 0.000032 0.000011
								CHANNELS 3 Zrotation Yrotation Xrotation
								End Site
								{
									OFFSET 0.000000 0.000000 0.000000
								}
							}
						}
					}
				}
				JOINT RightShoulder
				{
					OFFSET 19.746101 -1.480358 -6.000078
					CHANNELS 3 Zrotation Yrotation Xrotation
					JOINT RightArm
					{
						OFFSET 11.284140 -0.000000 -0.000001
						CHANNELS 3 Zrotation Yrotation Xrotation
						JOINT RightForeArm
						{
							OFFSET 33.000103 0.000016 -0.000001
							CHANNELS 3 Zrotation Yrotation Xrotation
							JOINT RightHand
							{
								OFFSET 25.199762 0.000123 0.000432
								CHANNELS 3 Zrotation Yrotation Xrotation
								End Site
								{
									OFFSET 0.000000 0.000000 0.000000
								}
							}
						}
					}
				}
			}
		}
	}
}
MOTION
Frames: 7184
Frame Time: 0.033333
-224.689499 91.882057 -431.625488 91.911438 5.797277 88.877003 -173.825528 -1.357308 177.800425 -8.189491 -1.405895 -8.436305 72.185442 4.507118 -5.155255 21.454553 0.003093 -0.000001 -171.432885 6.949201 -171.416698 -10.045297 3.093132 10.020026 77.536437 -5.597355 6.281055 21.454560 -0.003116 -0.000019 6.938949 -0.425479 0.083695 3.863257 -0.856897 0.159520 3.545122 -0.855727 0.162024 5.062785 -5.967228 -2.955440 -18.981926 10.359066 -7.856502 -95.384773 -86.109622 -86.694348 -1.744817 13.837001 -10.454342 -23.351689 -15.926113 1.189659 8.269972 -3.007814 30.896924 -101.166918 82.060809 81.176272 -5.484384 -12.039369 6.737670 -22.244692 15.926982 -1.200329 7.052535 3.369541 -14.328745 
...
```

**注意：在身体关节中，Spine1的子关节必须有Spine2，若无，需进行额外的映射，目前待完成（2025年8月11日）；MOITION下每一行代表一帧的关节数据，长度应与上述关节数量一致，每一个数据之间用一个空格隔开，最后一个数据后直接换行，不需要额外的空格或其他字符**

确定格式后，使用下列代码进行重定向

```bash
python scripts/bvh_to_robot.py --bvh_file <PATH_TO_YOUR_FILE>.bvh --robot unitree_g1 --save_path <PATH_TO_YOUR_OUTPUT_FILE>.pkl
```

#### 参数调整

GMR是基于mink的重定向项目，因此参数的修改与PBHC类似

文件motion_retarget.py	路径GMR/general_motion_retargeting

```bash
#行80
		for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []
```

该项目使用二次逆运动学，具体关节可见目录GMR/general_motion_retargeting/ik_configs下<数据格式>\_to_<机器人名称>.json中，该文件同时也存放关节旋转角度，关节全局位置、旋转权重和逆运动学组，若有需要自行更改。

#### 可视化

完成重定向后，使用以下代码可视化

```bash
python scripts/vis_robot_motion.py --robot unitree_g1 --robot_motion_path <PATH_TO_YOUR_OUTOUT_FILE>.pkl
```

#### 