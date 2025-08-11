# 开源重定向项目-GMR

**项目地址：**https://github.com/YanjieZe/GMR

**项目优势：**简单易部署，兼容SMPL-X(\*.npz)与Lafan(*.bvh)的原始数据格式，**关节转动效果较好**，脚趾与地板接触良好。

**项目缺点：**暂时没有遇到



## 部署

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



## 重定向（基于SMPL/SMPL-X）

本项目天然支持29自由度的g1机器人重定向，基于SMPL-X数据集，故若需要使用SMPL数据集，需要进行如下修改，在保留SMPL-X格式的同时，兼容SMPL。

**创建目录GMR/assert/body_models/smplx/**

```bash
GMR/assert/body_models/smplx/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_FEMALE.pkl
└──SMPLX_MALE.pkl
#建议一并将*.npz文件一同放入，保证最大兼容性
```

完成上述内容后，已经可以正常使用SMPL-X数据集（如AMASS）进行重定向，若需要使用SMPL格式（如GVHMR产出的），需要进行一些修改

**文件smpl.py	路径GMR/general_motion_retargeting/utils**

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



## 重定向（基于LaFan/动作捕捉）

**注意：由于各家动作捕捉生成的\*.bvh格式不统一，本项目基于LaFan格式编写，如果需要，请先进行文件格式转换，格式参照标准LaFan数据格式（详见data.md）**

**注意：在身体关节中，Spine1的子关节必须有Spine2，若无，需进行额外的映射，目前待完成（2025年8月11日）；MOITION下每一行代表一帧的关节数据，长度应与上述关节数量一致，每一个数据之间用一个空格隔开，最后一个数据后直接换行，不需要额外的空格或其他字符**

确定格式后，使用下列代码进行重定向

```bash
python scripts/bvh_to_robot.py --bvh_file <PATH_TO_YOUR_FILE>.bvh --robot unitree_g1 --save_path <PATH_TO_YOUR_OUTPUT_FILE>.pkl
```

更多参数详见GMR/README.md



## 重定向的参数与调整

GMR是基于mink的重定向项目，因此参数的修改与PBHC类似

### 全局参数

每一种重定向都有其对应的.json文件，在路径GMR/general_motion_retargeting/ik_configs下，以SMPL/SMPL-X到unitree_g1的重定向为例，参数保存在GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json中

```bash
{
    "robot_root_name": "pelvis",
    "human_root_name": "pelvis",
    "ground_height": 0.0,
    "human_height_assumption": 1.8,
    "use_ik_match_table1": true,
    "use_ik_match_table2": true,
    "human_scale_table": {
        "pelvis": 0.9,
        "spine3": 0.9,
		...
    },

    "ik_match_table1": {
        "pelvis": [
            "pelvis",
            100,
            10,
            [
                0.0,
                0.0,
                0.0
            ],
            [
                0.5,
                -0.5,
                -0.5,
                -0.5
            ]
        ],
        ...
    },
   "ik_match_table2": {
        "pelvis": [
            "pelvis",
            100,
            5,
            [
                0.0,
                0.0,
                0.0
            ],
            [
                0.5,
                -0.5,
                -0.5,
                -0.5
            ]
        ],
        ...
    }
}
```

对于ik_match_table中的数据解释：

```bash
"<G1 机器人关节名>": [
    "<源骨架关节名>",
    <位置权重>,
    <旋转权重>,
    [<平移偏移量 x, y, z>],
    [<旋转偏移量 x, y, z, w>]
]
```

### 任务设置

**文件motion_retarget.py	路径GMR/general_motion_retargeting**

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



## 可视化

完成重定向后，使用以下代码可视化

```bash
python scripts/vis_robot_motion.py --robot unitree_g1 --robot_motion_path <PATH_TO_YOUR_OUTOUT_FILE>.pkl
```

#### 