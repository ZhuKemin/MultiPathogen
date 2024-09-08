import subprocess
import pandas as pd
from io import StringIO
import os
from itertools import product

class DecompositionModel:
    def __init__(self):
        """
        初始化DecompositionModel类，固定R脚本的路径。
        """
        self.stl_script_path = os.path.join(os.path.dirname(__file__), "stl.R")
        self.mstl_script_path = os.path.join(os.path.dirname(__file__), "mstl.R")

    def decompose(self, ts_data, periods):
        """
        根据 periods 的长度决定调用STL或MSTL分解。
        :param ts_data: 时间序列数据，pandas DataFrame格式，包含'date'和'value'列
        :param periods: 包含周期长度的列表或元组
        :return: 分解后的结果，字典格式，键为周期组合，值为pandas DataFrame格式的分解结果
        """
        # 直接执行STL或MSTL
        if len(periods) == 1:
            command = ["Rscript", self.stl_script_path, str(periods[0])]
        else:
            command = ["Rscript", self.mstl_script_path, *map(str, periods)]
        result = self._run_r_script(command, ts_data)
        return result

    def _run_r_script(self, command, ts_data):
        """
        运行给定的R脚本并返回结果。
        :param command: 运行R脚本的命令
        :param ts_data: 要传递给R脚本的数据，pandas DataFrame格式
        :return: 脚本的输出，pandas DataFrame格式
        """
        ts_csv = ts_data.to_csv(index=False)

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input=ts_csv)

        if stderr:
            print(f"Error in R script: {stderr}")

        return pd.read_csv(StringIO(stdout))



# TODO: 待删除的测试内容
# ts_data = pd.read_csv('../../data/processed/flu/Chen-2023-Xian-Cases.csv')
# DM = DecompositionModel()
# result_df = DM.decompose(ts_data, periods=[12, 44])
# print (result_df)
# exit()
