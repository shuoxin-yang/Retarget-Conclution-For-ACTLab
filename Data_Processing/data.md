# 文件格式与基础数据处理（\*.npz/\*.bvh/\*.csv）

针对不同的源文件，下述重定向具有不同的使用方式和修改内容。

## (1)SMPL/SMPL-X(*.npz)

SMPL-X为SMPL的拓展版本，**SMPL-X格式天然支持SMPL重定向<u>（帧率除外）</u>**，且SMPL-X相比于SMPL具有更多的关节自由度。

\*.npz文件为压缩文件，内部包含多个\*.npy文件，可直接使用归档管理器查看内部构造

**SMPL格式**

```bash
*.npz/
├── betas.npy
├── gender.npy
├── mocap_framerate.npy
├── poses.npy
└── trans.npy
```

**SMPL-X格式**

```bash
*.npz/
├── betas.npy
├── gender.npy
├── latent_labels.npy
├── makers_latent.npy
├── makers_latent_vids.npy
├── mocap_frame_rate.npy
├── mocap_time_length.npy
├── num_betas.npy
├── pose_body.npy
├── pose_eye.npy
├── pose_hand.npy
├── pose_jaw.npy
├── poses.npy
├── root_orient.npy
├── surface_model_type.npy
└── trans.npy
```

以上为二者详细内容

### 原生基于SMPL-X的重定向

关节数据(pose_body or pose_aa)来自*.npz/pose_body.npy (N,63)，全局转动(global_orient)来自\*.npz/root_orient.npy (N,3)，帧率(fps or mocap_frame_rate)来自\*.npz/mocap_frame_rate.npy

### 原生基于SMPL的重定向

关节数据(pose_body or pose_aa)来自\*.npz/poses.npy的后69维(N,3:72)，全局转动来自\*.npz/poses.npy的前三维(N,0:3)，帧率(fps or mocap_framerate)来自\*.npz/mocap_framerate.npy

### 将基于SMPL映射到SMPL-X：

```bash
#通用映射，具体实施略有不同
smplx_data["global_orient"] = smpl_data["poses"][:,:3]#映射全局转动
smplx_data["pose_body"] = smpl_data["poses"][:,3:66]#映射主要关节
smplx_data["mocap_frame_rate"] = smpl_data["mocap_framerate"]#映射帧率
#important#
#smpl_data["poses"][:,66:72]为L_Hand/R_Hand，在本次重定向不考虑，对应smplx_data["pose_hand"]
```

## (2)动作捕捉(\*.bvh)

PBHC不支持.bvh的重定向，故以下分析基于GMR(GeneralMotionRetargeting)。

**\*.bvh为文本文档格式，可直接使用VSCode阅读。**

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

以上为一个标准的.bvh文件在文本阅读器里面的展示，joint可以嵌套，frame_data以单一空格区分，换行为一帧完结，**行尾不包含任何除换行符(\n)以外的任何字符，包括空格。**

标准LaFan数据格式

以下给出一个标准LaFan的数据样例，MOTION部分有所删减，仅供参考

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

## (3)训练文件(\*.csv)

本文件是将重定向结果(\*.pkl)转换为训练数据(\*.csv)以进一步训练策略

\*.csv文件为文本文档，可直接使用VSCode阅读

文件只包含29关节的关节数据，顺序依照如下列表

```bash
G1: (30 FPS)
    root_joint(X Y Z QX QY QZ QW)
    left_hip_pitch_joint
    left_hip_roll_joint
    left_hip_yaw_joint
    left_knee_joint
    left_ankle_pitch_joint
    left_ankle_roll_joint
    right_hip_pitch_joint
    right_hip_roll_joint
    right_hip_yaw_joint
    right_knee_joint
    right_ankle_pitch_joint
    right_ankle_roll_joint
    waist_yaw_joint
    waist_roll_joint
    waist_pitch_joint
    left_shoulder_pitch_joint
    left_shoulder_roll_joint
    left_shoulder_yaw_joint
    left_elbow_joint
    left_wrist_roll_joint
    left_wrist_pitch_joint
    left_wrist_yaw_joint
    right_shoulder_pitch_joint
    right_shoulder_roll_joint
    right_shoulder_yaw_joint
    right_elbow_joint
    right_wrist_roll_joint
    right_wrist_pitch_joint
    right_wrist_yaw_joint
```

**\*.csv文件每一行为一帧（frame），数据之间以空格相隔，行尾仅存在换行符（\n）**

文件夹中给出了基于PBHC的输出文件（\*.pkl）到\*.csv的转换器：PBHC_to_csv.py