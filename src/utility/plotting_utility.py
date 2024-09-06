import sys
import os

# 获取当前文件的路径，并设置项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)

from src.visualization.decomposition_visualizer import DecompositionVisualizer
import pandas as pd
import numpy as np



class PlottingUtility:
    def __init__(self, base_dir):
        """
        初始化PlottingUtility类。
        
        :param base_dir: 实验结果的基础路径。
        """
        self.base_dir = base_dir
    
    def load_projection_results(self, city, pathogen, scenario="cross_year"):
        """
        加载指定城市和病原体的投影结果。
        
        :param city: 城市名称。
        :param pathogen: 病原体名称。
        :param scenario: 场景名称，默认为 "cross_year"。
        :return: 包含投影数据的字典，key为周期组合，value为DataFrame。
        """
        projection_dir = os.path.join(self.base_dir, f"{city}-{pathogen}", scenario, "projection")
        projection_files = [f for f in os.listdir(projection_dir) if f.endswith(".csv")]
        
        projection_results = {}
        for file_name in projection_files:
            period_combination = tuple(map(int, file_name.replace('cycle_', '').replace('.csv', '').split('_')))
            df = pd.read_csv(os.path.join(projection_dir, file_name))
            df['date'] = pd.to_datetime(df['date'])
            projection_results[period_combination] = df
        
        return projection_results

    def select_top_cycles(self, city, pathogen, metric='acf', percent=10, scenario="cross_year"):
        """
        从metrics.csv中选择指定百分比的最优周期组合。
        
        :param city: 城市名称。
        :param pathogen: 病原体名称。
        :param metric: 用于筛选的指标名称，默认为 'acf'。
        :param percent: 选择的百分比，默认为10%。
        :param scenario: 场景名称，默认为 "cross_year"。
        :return: 包含选定周期组合及其指标的DataFrame。
        """
        metrics_path = os.path.join(self.base_dir, f"{city}-{pathogen}", scenario, "metrics", "metrics.csv")
        metrics_df = pd.read_csv(metrics_path)
        
        sorted_df = metrics_df.sort_values(by=metric, ascending=True)
        top_n = int(len(sorted_df) * (percent / 100))
        selected_cycles_df = sorted_df.head(top_n)
        
        return selected_cycles_df

    def plot_selected_projections(self, city, pathogen, metric='acf', percent=10, scenario="cross_year"):
        """
        绘制指定城市和病原体的选定周期组合的投影图。
        
        :param city: 城市名称。
        :param pathogen: 病原体名称。
        :param metric: 用于筛选的指标名称，默认为 'acf'。
        :param percent: 选择的百分比，默认为10%。
        :param scenario: 场景名称，默认为 "cross_year"。
        """
        # 获取选定的周期组合
        selected_cycles_df = self.select_top_cycles(city, pathogen, metric, percent, scenario)
        
        # 加载投影结果
        projection_results = self.load_projection_results(city, pathogen, scenario)
        
        # 可视化投影结果
        output_dir = os.path.join(self.base_dir, f"{city}-{pathogen}", scenario, "figures")
        visualizer = DecompositionVisualizer(output_dir=output_dir)
        visualizer.plot_multiple_projections(selected_cycles_df, projection_results, title=f"{city} - {pathogen} Projection")


    def load_metrics_df(self, city, pathogen, scenario="cross_year"):
        """
        加载指定城市和病原体的 metrics DataFrame。
        """
        metrics_path = os.path.join(self.base_dir, f"{city}-{pathogen}", scenario, "metrics", "metrics.csv")
        return pd.read_csv(metrics_path)

    def _calculate_combined_scores(self, metrics_df_flu, metrics_df_rsv, score_column='acf'):
        """
        计算每对周期组合的综合得分，并过滤掉周期相等的组合。
        """
        # 过滤掉两个周期相等的组合
        metrics_df_flu_filtered = metrics_df_flu[metrics_df_flu['cycle1'] != metrics_df_flu['cycle2']]
        metrics_df_rsv_filtered = metrics_df_rsv[metrics_df_rsv['cycle1'] != metrics_df_rsv['cycle2']]

        flu_scores = metrics_df_flu_filtered[score_column].values[:, np.newaxis]
        rsv_scores = metrics_df_rsv_filtered[score_column].values[np.newaxis, :]
        combined_scores = flu_scores + rsv_scores

        combined_scores_df = pd.DataFrame({
            'cycle_flu': np.repeat(metrics_df_flu_filtered['cycle1'].astype(str) + '_' + metrics_df_flu_filtered['cycle2'].astype(str), len(metrics_df_rsv_filtered)),
            'cycle_rsv': np.tile(metrics_df_rsv_filtered['cycle1'].astype(str) + '_' + metrics_df_rsv_filtered['cycle2'].astype(str), len(metrics_df_flu_filtered)),
            'combined_score': combined_scores.ravel()
        })

        return combined_scores_df

    def _select_top_combinations(self, combined_scores_df, top_percent=5):
        """
        选择得分最高的前5%组合。
        """
        combined_scores_df = combined_scores_df.sort_values(by='combined_score', ascending=True)
        top_n = int(len(combined_scores_df) * (top_percent / 100))
        top_combinations_df = combined_scores_df.head(top_n)
        return top_combinations_df

    def plot_combined_top_projections_for_city(self, city, metric='acf', percent=5):
        """
        对指定城市进行组合得分计算并绘制叠加投影图。
        """
        metrics_df_flu = self.load_metrics_df(city, 'flu')
        metrics_df_rsv = self.load_metrics_df(city, 'rsv')
        combined_scores_df = self._calculate_combined_scores(metrics_df_flu, metrics_df_rsv, score_column=metric)
        top_combinations_df = self._select_top_combinations(combined_scores_df, top_percent=percent)
        visualizer = DecompositionVisualizer(output_dir=os.path.join(self.base_dir, f"{city}-combined", "figures"))
        visualizer.plot_combined_top_projections(city=city, base_dir=self.base_dir, top_combinations_df=top_combinations_df)



# 调用示例
if __name__ == "__main__":
    base_dir = 'E:/MultiPathogen/results'
    plotter = PlottingUtility(base_dir)
    
    # plotter.plot_selected_projections(city="Xian", pathogen="flu")

    # plotter.plot_combined_top_projections_for_city(city="Lanzhou", metric='acf', percent=0.05)
    plotter.plot_combined_top_projections_for_city(city="Yunfu", metric='acf', percent=3)

