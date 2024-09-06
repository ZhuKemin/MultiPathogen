import os
import pandas as pd
import matplotlib.pyplot as plt

class NLLPlotter:
    def __init__(self, city_list, pathogen_list, metric='nll'):
        """
        初始化类
        :param city_list: 城市名称列表
        :param pathogen_list: 病种名称列表
        :param metric: 要选取的指标，默认为 'nll'
        """
        self.city_list = city_list
        self.pathogen_list = pathogen_list
        self.metric = metric

    def _get_best_metric_value(self, filepath):
        """
        读取 metrics.csv 文件并返回最小的 nll 值
        :param filepath: metrics.csv 文件的路径
        :return: 指标的最小值
        """
        df = pd.read_csv(filepath)
        if self.metric in df.columns:
            return df[self.metric].min()
        else:
            raise ValueError(f"未找到指标 '{self.metric}' 列")

    def plot_best_nll(self, base_dir):
        """
        绘制城市-病种在 Baseline 和 cross-year 场景下的 nll 最佳值
        :param base_dir: 数据的基础路径
        """
        x_values = []
        y_values = []
        colors = []
        labels = []
        
        for city in self.city_list:
            for pathogen in self.pathogen_list:
                baseline_file = os.path.join(base_dir, f"{city}-{pathogen}", "baseline", "metrics", "metrics.csv")
                crossyear_file = os.path.join(base_dir, f"{city}-{pathogen}", "cross_year", "metrics", "metrics.csv")

                baseline_value = self._get_best_metric_value(baseline_file)
                crossyear_value = self._get_best_metric_value(crossyear_file)

                x_values.append(crossyear_value)
                y_values.append(baseline_value)
                labels.append(f"{city}-{pathogen}")
                
                # 根据病种选择颜色
                if pathogen == "flu":
                    colors.append('tab:blue')
                elif pathogen == "rsv":
                    colors.append('tab:red')
        
        # 设置字体为 Times New Roman 并调整字体大小
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 14

        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color=colors, alpha=0.7)

        plt.xscale('log')  # 设置 x 轴为对数刻度
        plt.yscale('log')  # 设置 y 轴为对数刻度

        # 设置等比例轴
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        # 设置最小和最大值
        minx, maxx = 1e1, 1e5

        # 设置 xlim 和 ylim
        plt.xlim(minx, maxx)
        plt.ylim(minx, maxx)

        # 添加 y=x 的黑色虚线
        plt.plot([minx, maxx], [minx, maxx], 'k--', lw=1)

        # 动态设置轴标签和标题
        plt.xlabel(f'Cross-Year {self.metric.upper()} (Log Scale)', fontsize=16)
        plt.ylabel(f'Baseline {self.metric.upper()} (Log Scale)', fontsize=16)
        plt.title(f'Best {self.metric.upper()} Comparison Between Cross-Year and Baseline', fontsize=18)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()

# 使用示例
city_list = ['Beijing', 'Guangzhou', 'Wuhan', 'Xian', 'Lanzhou', 'Suzhou', 'Wenzhou', 'Yunfu']
pathogen_list = ["flu", "rsv"]
plotter = NLLPlotter(city_list, pathogen_list, metric="nll")  # 动态传入指标名称
plotter.plot_best_nll(base_dir="E:\\MultiPathogen\\results")
