import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from pandas.tseries.offsets import DateOffset

# 设置全局字体为Times New Roman，增大字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22  # 增大字体大小

# 城市列表（拼音）
cities = ['Beijing', 'Guangzhou', 'Wuhan', 'Xian', 'Lanzhou', 'Suzhou', 'Wenzhou', 'Yunfu']

# 数据路径
flu_path = '../../data/processed/flu'
rsv_path = '../../data/processed/rsv'

def find_city_file(city, directory):
    pattern = re.compile(rf'^[^-]+-[^-]+-{city}-Cases\.csv$')
    for filename in os.listdir(directory):
        if pattern.match(filename):
            return os.path.join(directory, filename)
    return None

def convert_date_format(df):
    sample_date = df['Year-Month'].iloc[0]
    if '-' in sample_date:
        try:
            if sample_date.split('-')[0].isdigit():
                df['Year-Month'] = pd.to_datetime(df['Year-Month'], format='%Y-%m')
            else:
                df['Year-Month'] = pd.to_datetime(df['Year-Month'], format='%b-%y')
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            return None
    else:
        print("Unexpected date format.")
        return None
    return df

def plot_city_sequences(city):
    # 查找符合城市名称的第一个文件
    flu_file = find_city_file(city, flu_path)
    rsv_file = find_city_file(city, rsv_path)

    if not flu_file or not rsv_file:
        print(f"Data for {city} not found in one of the directories.")
        return

    # 读取数据
    flu_df = pd.read_csv(flu_file)
    rsv_df = pd.read_csv(rsv_file)

    # 转换日期格式
    flu_df = convert_date_format(flu_df)
    rsv_df = convert_date_format(rsv_df)

    if flu_df is None or rsv_df is None:
        print(f"Skipping {city} due to date parsing issues.")
        return

    # 合并两个数据集的日期范围并扩展6个月
    combined_dates = pd.to_datetime(sorted(set(flu_df['Year-Month']).union(set(rsv_df['Year-Month']))))
    extended_start = combined_dates.min() - DateOffset(months=6)
    extended_end = combined_dates.max() + DateOffset(months=6)
    full_date_range = pd.date_range(start=extended_start, end=extended_end, freq='M')

    # 创建绘图
    fig, ax1 = plt.subplots(figsize=(9, 6))  # 增大图形尺寸

    # 绘制flu的折线图
    ax1.plot(flu_df['Year-Month'], flu_df['Positive Cases'], color='tab:green', label='Flu',
             linewidth=2, marker='o')  # 增加线宽和标记
    ax1.set_ylabel('Positive Cases (Flu)', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:cyan')

    # 添加第二个y轴绘制rsv的折线图
    ax2 = ax1.twinx()
    ax2.plot(rsv_df['Year-Month'], rsv_df['Positive Cases'], color='tab:orange', label='RSV',
             linewidth=2, marker='s')  # 增加线宽和标记
    ax2.set_ylabel('Positive Cases (RSV)', color='tab:orange', rotation=270, labelpad=20)  # 旋转270度并增加label与轴的距离
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 设置x轴标签
    # ax1.set_xlabel('Year-Month', labelpad=5)
    
    # 只显示1月的x轴刻度标签
    ax1.set_xticks([date for date in full_date_range if (date.month == 1)&(date.year%2 == 1)])
    ax1.set_xticklabels([date.strftime('%Y-%m') for date in full_date_range if (date.month == 1)&(date.year%2 == 1)])

    # 添加网格
    ax1.grid(True, linestyle='-.', color='lightgrey', axis='both')

    # 设置标题仅显示城市名
    plt.title(f'{city}', fontsize=36)

    # 保存图像
    plt.savefig(f'../../reports/figures/{city}_disease_sequence.png', dpi=300)
    plt.close()

# 遍历每个城市并绘图
for city in cities:
    plot_city_sequences(city)

print("Disease sequence plots saved.")
