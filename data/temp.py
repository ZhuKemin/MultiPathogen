import pandas as pd
import yaml

def convert_xlsx_to_yaml(xlsx_file, yaml_file):
    try:
        # 读取 Excel 文件的单个工作表
        df = pd.read_excel(xlsx_file)

        # 将日期列转换为字符串格式
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')
        
        # 创建一个空字典用于存储城市和病原体的层次化数据
        city_pathogen_data = {}

        # 遍历每一行数据，并将其组织为 YAML 格式
        for _, row in df.iterrows():
            city = row['city']
            pathogen = row['Pathogen']
            
            # 初始化城市的键，如果不存在
            if city not in city_pathogen_data:
                city_pathogen_data[city] = {}
            
            # 初始化病原体的键，如果不存在
            if pathogen not in city_pathogen_data[city]:
                city_pathogen_data[city][pathogen] = {
                    'name': row['name'],
                    'raw_data_freq': row['raw_data_freq'],
                    'proc_data_freq': row['proc_data_freq'],
                    'data_type': row['data_type'],
                    'start_date': row['start_date']
                }

        # 将字典写入 YAML 文件
        with open(yaml_file, 'w') as f:
            yaml.dump(city_pathogen_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Successfully converted {xlsx_file} to {yaml_file}")
    
    except Exception as e:
        print(f"Failed to convert {xlsx_file} to YAML: {e}")

if __name__ == "__main__":
    # 设置要转换的文件名
    xlsx_file = "metadata.xlsx"
    yaml_file = "metadata.yaml"
    
    # 执行转换
    convert_xlsx_to_yaml(xlsx_file, yaml_file)
