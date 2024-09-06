import os
import pandas as pd
import numpy as np

class ProjectionStatistics:
    def __init__(self, results_dir, output_dir="statistics_output"):
        """
        初始化ProjectionStatistics类。
        
        :param results_dir: 结果文件夹的路径，包含所有城市-病原体组合的投影结果。
        :param output_dir: 保存统计结果的输出目录，默认为'statistics_output'。
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_metrics(self, city_pathogen_dir):
        """
        加载metrics.csv文件，并筛选出acf最小的10%周期组合。
        
        :param city_pathogen_dir: 城市-病原体组合的目录路径。
        :return: 包含top 10%周期组合的DataFrame。
        """
        metrics_path = os.path.join(city_pathogen_dir, "cross_year", "metrics", "metrics.csv")
        metrics_df = pd.read_csv(metrics_path)

        # 筛选acf最小的10%周期组合
        top_10_percent_df = metrics_df.nsmallest(int(len(metrics_df) * 0.1), 'acf')
        
        return top_10_percent_df
    
    def calculate_duration(self, projection_df):
        """
        计算持续时间，即预测序列中连续超过最大值10%的数据点的数目。
        
        :param projection_df: 投影结果的DataFrame。
        :return: 持续时间（连续超过10%最大值的数据点数）。
        """
        projected_values = projection_df[projection_df['is_projected']]['projection']
        max_value = projected_values.max()
        threshold = 0.1 * max_value
        
        # 计算连续超过阈值的长度
        duration = (projected_values >= threshold).astype(int).groupby(projected_values.lt(threshold).cumsum()).sum().max()
        
        return duration
    
    def calculate_time_to_peak(self, projection_df):
        """
        计算第一次高峰到来的时间，即超过最大值80%的第一个在2024以后的时间点。
        
        :param projection_df: 投影结果的DataFrame。
        :return: 第一次高峰到来的时间（时间戳）。
        """
        projected_values = projection_df[projection_df['is_projected']]['projection']
        max_value = projected_values.max()
        threshold = 0.95 * max_value
        
        # 找到超过阈值的第一个时间点
        peak_times = projection_df[(projection_df['is_projected']) & (projection_df['projection'] >= threshold) & 
                                   (projection_df['date'] > '2025-01-01')]['date']
        
        if not peak_times.empty:
            return peak_times.iloc[0]
        else:
            return None
    
    def calculate_severity(self, projection_df):
        """
        计算严重程度，即最大值相较于平均值的比例。
        
        :param projection_df: 投影结果的DataFrame。
        :return: 严重程度（最大值/平均值）。
        """
        projected_values = projection_df[projection_df['is_projected']]['projection']
        max_value = projected_values.max()
        mean_value = projected_values.mean()
        
        severity = max_value / mean_value
        return severity
    
    def process_city_pathogen(self, city_pathogen_dir):
        """
        处理单个城市-病原体组合的所有周期组合，计算统计指标。
        
        :param city_pathogen_dir: 城市-病原体组合的目录路径。
        :return: 包含统计指标（最大值、最小值和中位数）的字典。
        """
        top_10_percent_df = self.load_metrics(city_pathogen_dir)

        durations = []
        time_to_peaks = []
        severities = []

        columns_cycle = [col for col in top_10_percent_df.columns if col.lower().startswith('cycle')]

        for _, row in top_10_percent_df.iterrows():
            period_combination = tuple(row[columns_cycle].astype(int))
            period_str = '_'.join(map(str, period_combination))
            
            projection_path = os.path.join(city_pathogen_dir, "cross_year", "projection", f"cycle_{period_str}.csv")
            projection_df = pd.read_csv(projection_path)
            projection_df['date'] = pd.to_datetime(projection_df['date'])
            
            durations.append(self.calculate_duration(projection_df))
            time_to_peaks.append(self.calculate_time_to_peak(projection_df))
            severities.append(self.calculate_severity(projection_df))
        
        # 过滤掉 None 值，避免在计算最大最小和中位数时出错
        time_to_peaks_filtered = [t.value for t in time_to_peaks if t is not None]  # 转换为纳秒表示的整数

        stats = {
            'City': city_pathogen_dir.split('\\')[-1].split('-')[0],
            'Pathogen': city_pathogen_dir.split('-')[1],
            'Duration': f"{np.median(durations)} ({np.min(durations)}, {np.max(durations)})",
            'Severity': f"{np.median(severities):.2f} ({np.min(severities):.2f}, {np.max(severities):.2f})",
            'Time_to_Peak': f"{pd.to_datetime(np.median(time_to_peaks_filtered)).strftime('%Y-%m')} ({pd.to_datetime(min(time_to_peaks_filtered)).strftime('%Y-%m')}, {pd.to_datetime(max(time_to_peaks_filtered)).strftime('%Y-%m')})" if time_to_peaks_filtered else None,
        }
        
        return stats
    
    def run(self):
        """
        运行统计模块，计算所有城市-病原体组合的统计指标，并保存结果。
        """
        results = []
        
        for city_pathogen_dir in os.listdir(self.results_dir):
            full_path = os.path.join(self.results_dir, city_pathogen_dir)
            if os.path.isdir(full_path):
                print(f"Processing {city_pathogen_dir}...")
                stats = self.process_city_pathogen(full_path)
                results.append(stats)
        
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.output_dir, "projection_statistics.xlsx")
        results_df.to_excel(output_path, index=False)
        print(f"Statistics saved to {output_path}")


if __name__ == '__main__':
    # 设置工作目录为项目的根目录
    os.chdir('E:/MultiPathogen')
    print(f"Current working directory: {os.getcwd()}")

    # 创建 ProjectionStatistics 实例
    statistics_calculator = ProjectionStatistics(results_dir='E:/MultiPathogen/results', output_dir='E:/MultiPathogen/results')
    
    # 计算并保存统计结果
    statistics_calculator.run()
