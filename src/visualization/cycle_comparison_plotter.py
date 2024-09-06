import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_cycle_comparisons(city, pathogen, base_dir, metric='acf', top_percent=5):
    """
    绘制某个城市的 RSV 和 flu 在 cross year 场景下的周期组合选择。

    :param city: 城市名称
    :param pathogen: 病原体名称
    :param base_dir: 数据所在的基础路径
    :param metric: 用于筛选的指标，默认为 'acf'
    :param top_percent: 选择的顶级周期组合的百分比，默认为10%
    """
    # 构造路径
    cross_year_dir = os.path.join(base_dir, f"{city}-{pathogen}", "cross_year")
    metrics_path = os.path.join(cross_year_dir, "metrics", "metrics.csv")
    
    # 读取metrics.csv
    metrics_df = pd.read_csv(metrics_path)
    
    # 筛选acf最小的top_percent%组合
    # top_combinations = metrics_df.nsmallest(int(len(metrics_df) * top_percent / 100), metric)

    # 筛选 ACF 最小的一个
    top_combinations = metrics_df.loc[[metrics_df[metric].idxmin()]]

    # 获取所有周期组合的文件路径
    all_files = [f for f in os.listdir(os.path.join(cross_year_dir, "decomposition")) if f.endswith(".csv")]
    
    # 设置绘图，动态生成子图
    sample_df = pd.read_csv(os.path.join(cross_year_dir, "decomposition", all_files[0]))
    seasonal_columns = [col for col in sample_df.columns if col.lower().startswith('seasonal')]
    num_seasonal_components = len(seasonal_columns)

    fig, axs = plt.subplots(num_seasonal_components, 1, figsize=(15, 3 * num_seasonal_components), sharex=True)
    axs = axs if num_seasonal_components > 1 else [axs]  # 确保 axs 是一个列表
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14  # 全局字体大小
    
    for file_name in all_files:
        # 读取周期组合文件
        df = pd.read_csv(os.path.join(cross_year_dir, "decomposition", file_name))
        df['date'] = pd.to_datetime(df['date'])
        dates = df['date']
        
        # 找出周期组合
        period_combination = tuple(map(int, file_name.replace('cycle_', '').replace('.csv', '').split('_')))
        
        # 确定颜色和线宽
        if period_combination in [tuple(row[['cycle1', 'cycle2']].dropna().astype(int)) for _, row in top_combinations.iterrows()]:
            color = 'tab:orange'
            lw = 1.5
            zorder = 11
        else:
            color = 'lightgray'
            lw = 0.3
            zorder = 10

        seasonal_columns = [col for col in df.columns if col.lower().startswith('seasonal')]

        # 绘制每个季节性成分到对应的子图上
        for i, col in enumerate(seasonal_columns):
            cycle_value = int(round(float(col.replace('seasonal', '')))) if col != 'seasonal' else 12
            axs[i].plot(dates, df[col].values, lw=lw, color=color, zorder=zorder, alpha=0.5)
            ylabel = 'First Cycle' if i==0 else 'Second Cycle'
            axs[i].set_ylabel(ylabel, fontsize=16)
            axs[i].grid(True)
    
    axs[-1].set_xlabel('Date', fontsize=16)
    plt.suptitle(f"{city} - {pathogen} Cycle Comparison-Top 1", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 调整布局以适应标题

    # 保存图像
    output_dir = os.path.join(base_dir, f"{city}-{pathogen}", "figures")
    os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, f"{city}_{pathogen}_cycle_comparison.png"), dpi=300)
    plt.show()

# 调用示例
base_dir = 'E:/MultiPathogen/results'
plot_cycle_comparisons(city="Suzhou", pathogen="flu", base_dir=base_dir)
