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
        
        # 将 DataFrame 转换为字典
        data = df.to_dict(orient='records')
        
        # 将字典写入 YAML 文件
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Successfully converted {xlsx_file} to {yaml_file}")
    
    except Exception as e:
        print(f"Failed to convert {xlsx_file} to YAML: {e}")


if __name__ == "__main__":
    # 设置要转换的文件名
    xlsx_file = "metadata.xlsx"
    yaml_file = "metadata.yaml"
    
    # 执行转换
    convert_xlsx_to_yaml(xlsx_file, yaml_file)
