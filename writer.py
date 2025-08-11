import joblib
import numpy as np
import pandas as pd

# 1. 路径
pkl_path = 'smpl_retarget/retargeted_motion_data/mink/Walk_B15_-_Walk_turn_around_stageii_Ncl.pkl'
csv_path = 'smpl_retarget/retargeted_motion_data/mink/Walk.csv'

# 2. 读取
data = joblib.load(pkl_path)
motion = data[list(data.keys())[0]]          
root = motion['root_trans_offset']           
root_rot = motion['root_rot']
dof  = motion['dof']                       

# 3. 生成csv
table = np.hstack([root, root_rot,dof])
np.savetxt(csv_path, table, delimiter=',',fmt='%.10f')
print(csv_path)