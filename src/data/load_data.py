import os
import pandas as pd
import yaml

def load_data(city=None, file_path=None, base_path=None, pathogen=None, metadata_path=None):
    """
    加载指定城市或文件的时间序列数据。
    :param city: 城市名称，用于自动匹配文件
    :param file_path: 完整的文件路径
    :param base_path: 数据文件的基础路径
    :param pathogen: 病原体类型，如 "flu" 或 "rsv"
    :param metadata_path: 元数据文件的路径
    :return: 加载的时间序列数据 (pandas DataFrame)
    """
    if file_path:
        data = pd.read_csv(file_path)
    elif city and base_path:
        # 查找匹配的文件
        data = None
        for filename in os.listdir(base_path):
            if city.lower() in filename.lower():
                data = pd.read_csv(os.path.join(base_path, filename))
                break
        if data is None:
            raise FileNotFoundError(f"在路径 {base_path} 下未找到与城市 {city} 相关的数据文件。")
    else:
        raise ValueError("必须提供文件路径或城市名称和基础路径。")

    # 如果提供了元数据路径，则加载元数据
    if metadata_path:
        metadata = load_metadata(metadata_path)
        # 根据元数据动态调整数据处理（如时间频率等）

    # 标准化列名为 date 和 value
    data.columns = ['date', 'value']

    # 数据验证与清洗
    validate_and_clean_data(data)

    return data

def load_metadata(metadata_path):
    """
    加载元数据。
    :param metadata_path: 元数据文件的路径
    :return: 元数据信息 (dict)
    """
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = yaml.safe_load(file)
    return metadata

def validate_and_clean_data(data):
    """
    验证数据完整性并进行必要的清洗处理。
    :param data: 加载的时间序列数据 (pandas DataFrame)
    """
    # 进行数据验证和清洗
    if 'date' not in data.columns or 'value' not in data.columns:
        raise ValueError("数据缺少必要的列：date 或 value")

    # 自定义解析日期格式
    def parse_date(date_str):
        return pd.to_datetime(date_str, format='%b-%y', errors='coerce')

    # 应用解析函数，并处理错误
    data['date'] = data['date'].apply(parse_date)
    data.dropna(subset=['date'], inplace=True)

    # 处理缺失值、标准化时间格式等
    data.dropna(inplace=True)
