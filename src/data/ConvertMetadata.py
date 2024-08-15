import pandas as pd
import yaml
import os

# 读取 Excel 文件
excel_file_path = '../../data/metadata.xlsx'  # 请根据你的实际路径修改
df = pd.read_excel(excel_file_path)

# 确保 start_date 字段为字符串格式 (YYYY-MM-DD)
df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')

# 将 DataFrame 转换为字典
metadata_dict = df.to_dict(orient='records')

# 构造最终的 YAML 数据结构
metadata_yaml = {'datasets': metadata_dict}

# 将字典转换为 YAML 格式字符串
yaml_file_path = '../../data/metadata.yaml'  # 目标 YAML 文件路径
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(metadata_yaml, yaml_file, sort_keys=False)

print(f"YAML file has been saved to {yaml_file_path}")
