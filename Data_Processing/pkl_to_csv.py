import joblib as jb
import numpy as np
import argparse
import os
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--pkl_path",   type=str, required=True, help="retargeting pkl path")
parser.add_argument("--csv_path",  type=str,   default="default", help="output csv path")
parser.add_argument("--force_remake", action="store_true", default=False, help="force remake output file")
opt = parser.parse_args()

def collect_pkl_names(file_name: str) -> list[str]:
    """返回子一级目录（或自身）中所有 .pkl 的纯文件名（不含后缀）。"""
    if file_name.lower().endswith('.pkl') and os.path.isfile(file_name):
        return [os.path.splitext(os.path.basename(file_name))[0]]

    pattern = os.path.join(file_name, '*.pkl')
    return [os.path.splitext(os.path.basename(p))[0] for p in glob(pattern)]

if __name__ == "__main__":
    pkl_path = opt.pkl_path
    csv_path = opt.csv_path

    """若 csv_path 为 "default"，则根据 pkl_path 自动生成。"""
    if csv_path == "default":
        file_name = collect_pkl_names(pkl_path)
        os.makedirs("./output", exist_ok=True)
        csv_path = []
        if file_name.__len__() == 1:
            csv_path.append("./output/" + file_name[0] )
        else:
            for i in file_name:
                csv_path.append("./output/" + i  )
    else:
        if not os.path.exists(csv_path):
            raise ValueError(f"Output path {csv_path} does not exist")
        
    """遍历"""
    for i in range(file_name.__len__()):
        data = jb.load(pkl_path+"/" + file_name[i] + ".pkl")

        """关键字格式判断"""
        if data.keys().__len__() == 1:
            print("Type = PBHC format", end="\t")
            type = "PBHC"
            motion = data[list(data.keys())[0]]
            root_pos = motion['root_trans_offset']
            root_rot = motion['root_rot']
            dof_pos  = motion['dof']
        elif data.keys().__len__() > 1:
            print("Type = GMR format", end="\t")
            type = "GMR"
            motion = data
            root_pos = motion["root_pos"]
            root_rot = motion["root_rot"]
            dof_pos  = motion["dof_pos"]
        else:
            print(data.keys())
            raise ValueError("Cannot recognize input pkl format")
        
        """写入"""
        table = np.hstack([root_pos, root_rot,dof_pos])
        csv_path[i] += "_" + type + ".csv"

        """强制重新生成判断"""
        if os.path.exists(csv_path[i]) and opt.force_remake:
            os.remove(csv_path[i])
            print(f"File {csv_path[i]} already exists, removed", end="\t")
        elif os.path.exists(csv_path[i]) and (not opt.force_remake):
            raise ValueError(f"File {csv_path[i]} already exists, use --force_remake to overwrite")
        
        np.savetxt(csv_path[i], table, delimiter=',',fmt='%.17g')
        print(f"CSV file: {csv_path[i]} generated successfully")


    print(f"Successfully converted {file_name.__len__()} pkl files to csv files")