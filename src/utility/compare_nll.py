# src/utility/compare_nll.py

import subprocess
import os
from calculate_nll_python import calculate_nll_from_csv

def calculate_nll_r(file_path, shape_param, scale_param):
    r_script_path = os.path.join(os.path.dirname(__file__), 'calculate_nll_r.R')
    result = subprocess.run(
        ['Rscript', r_script_path, file_path, str(shape_param), str(scale_param)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())

def compare_nll(file_path, shape_param=1.0, scale_param=1.0):
    # 调用 Python 版本的 NLL 计算
    python_nll = calculate_nll_from_csv(file_path, shape_param, scale_param)

    # 调用 R 版本的 NLL 计算
    r_nll = calculate_nll_r(file_path, shape_param, scale_param)

    print(f"Python 计算的 NLL: {python_nll}")
    print(f"R 计算的 NLL: {r_nll}")
    print(f"差异: {abs(python_nll - r_nll)}")

if __name__ == "__main__":
    # 假设用户输入文件路径
    file_path = '../../results/Beijing-flu/cross_year/decomposition/cycle_14_30.csv'
    compare_nll(file_path, shape_param=1.0, scale_param=1.0)
