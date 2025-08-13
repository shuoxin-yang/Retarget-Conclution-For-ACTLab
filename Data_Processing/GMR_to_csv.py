import joblib
import numpy as np

# 1. 路径
pkl_path = 'motion_data/output/box.pkl'
csv_path = 'motion_data/output/box.csv'

# 2. 读取
motion = joblib.load(pkl_path)        
root_pos=motion["root_pos"]  # (N, 3)
root_rot=motion["root_rot"]  # (N, 4)
dof_pos=motion["dof_pos"]    # (N, 29)               

# 3. 生成csv
table = np.hstack([root_pos, root_rot,dof_pos])
np.savetxt(csv_path, table, delimiter=',',fmt="%.10f")
print(csv_path)
