import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import ccf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import os

class DecompositionVisualizer:
    def __init__(self, output_dir, max_lag_weeks=52):
        """
        初始化 DecompositionVisualizer。
        :param output_dir: 保存图像的输出目录。
        :param max_lag_weeks: 最大滞后周数，默认设置为52周。
        """
        self.output_dir = output_dir
        self.max_lag_weeks = max_lag_weeks  # 设置最大滞后周数
        plt.rcParams["font.family"] = "Times New Roman"

    def plot_decomposition_results(self, decomposition_results):
        """
        绘制所有周期组合的分解结果图。
        :param decomposition_results: 所有分解结果的字典，键为周期组合，值为DataFrame
        """
        # 计算全局Y轴限制
        self.global_y_limits = self._calculate_global_y_limits(decomposition_results)
        
        for period_combination, df in decomposition_results.items():
            self._plot_decomposition(period_combination, df)

    def _calculate_global_y_limits(self, decomposition_results):
        """
        计算所有分解结果中每个分量（趋势、季节性、残差）的全局最小值和最大值。
        :param decomposition_results: 所有分解结果的字典，键为周期组合，值为DataFrame
        :return: 包含每个分量全局最小值和最大值的字典
        """
        y_limits = {'trend': [np.inf, -np.inf], 'remainder': [np.inf, -np.inf], 'observed': [np.inf, -np.inf], 'seasonal': [np.inf, -np.inf]}

        for df in decomposition_results.values():
            df.columns = df.columns.str.lower()
            for key in y_limits.keys():
                if key == 'observed':
                    # 计算观测值：趋势 + 所有季节性 + 残差
                    seasonal_columns = [col for col in df.columns if col.startswith('seasonal')]
                    observed = df['trend'] + df[seasonal_columns].sum(axis=1) + df['remainder']
                    y_limits[key][0] = min(y_limits[key][0], observed.min())
                    y_limits[key][1] = max(y_limits[key][1], observed.max())
                else:
                    columns = [col for col in df.columns if col.startswith(key)]
                    for col in columns:
                        y_limits[key][0] = min(y_limits[key][0], df[col].min())
                        y_limits[key][1] = max(y_limits[key][1], df[col].max())

        # 扩展范围10%
        for key in y_limits.keys():
            y_range = y_limits[key][1] - y_limits[key][0]
            y_limits[key][0] -= 0.1 * y_range
            y_limits[key][1] += 0.1 * y_range

        return y_limits

    def _plot_decomposition(self, period_combination, df):
        """
        绘制单个周期组合的分解结果图。
        :param period_combination: 周期组合，元组形式
        :param df: 包含分解结果的DataFrame
        """
        dates = pd.to_datetime(df['date'].values)
        trend = df['trend'].values
        seasonal_columns = [col for col in df.columns if col.startswith('seasonal')]
        resid = df['remainder'].values
        observed = trend + df[seasonal_columns].sum(axis=1) + resid
        predict = trend + df[seasonal_columns].sum(axis=1)

        num_axes = 4 + len(seasonal_columns)  # 观测值vs预测值, 趋势, 各季节性, 残差, 自相关
        fig, axs = plt.subplots(num_axes, 1, figsize=(15, 2 * num_axes), sharex=False)

        # Plot observation vs. prediction
        ax1 = axs[0]
        ax1.plot(dates, observed, lw=2, color='tab:blue', label='Observed')
        ax1.plot(dates, predict, lw=1, ls='--', color='tab:red', label='Predicted')
        ax1.set_ylabel('Observed', fontsize=18)
        # ax1.legend(fontsize=12)  # Legend removed as per request
        ax1.set_ylim(self.global_y_limits['observed'])

        # Plot trend
        ax_trend = axs[1]
        ax_trend.plot(dates, trend, lw=2, color='tab:green')
        ax_trend.set_ylabel('Trend', fontsize=18)
        ax_trend.set_ylim(self.global_y_limits['trend'])

        # Plot seasonal components
        for i, col in enumerate(seasonal_columns):
            ax_seasonal = axs[2 + i]
            cycle_value = int(col.replace('seasonal', ''))  # 提取周期值
            ax_seasonal.plot(dates, df[col].values, lw=2, color='tab:orange')
            ax_seasonal.set_ylabel(f'Seasonal {cycle_value}', fontsize=18)
            ax_seasonal.set_ylim(self.global_y_limits['seasonal'])

            # 添加辅助线
            start_date = dates[0]
            while start_date <= dates[-1]:
                ax_seasonal.axvline(x=start_date, color='darkgray', linestyle='--')
                start_date += pd.DateOffset(weeks=cycle_value)  # 根据周期大小调整间隔

        # Plot remainders
        ax_resid = axs[2 + len(seasonal_columns)]
        ax_resid.scatter(dates, resid, color='tab:red')
        ax_resid.set_ylabel('Residual', fontsize=18)
        ax_resid.set_ylim(self.global_y_limits['remainder'])

        # Calculate and plot auto-correlation with lags
        correlations = ccf(resid, resid, adjusted=True)
        lags = np.arange(-self.max_lag_weeks, self.max_lag_weeks + 1)  # Use max_lag_weeks for dynamic lag range
        correlations_combined = np.concatenate((correlations[1:self.max_lag_weeks + 1][::-1], correlations[:self.max_lag_weeks + 1]))

        ax_autocorr = axs[-1]
        ax_autocorr.stem(lags, correlations_combined, linefmt='-', markerfmt='o', basefmt='r-')
        ax_autocorr.grid(which='both', linestyle='-.', linewidth=0.5, color='lightgrey')
        ax_autocorr.set_ylabel('Correlation', fontsize=18)
        ax_autocorr.set_xlim(-self.max_lag_weeks, self.max_lag_weeks)  # Set xlim dynamically based on max_lag_weeks

        # Apply formatting and save the plot
        for ax in axs[:-1]:  # Exclude the autocorrelation axis from date formatting
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.grid(which='both', linestyle='-.', linewidth=0.4, color='lightgrey')
            ax.set_axisbelow(True)
            ax.set_xlim(dates[0], dates[-1])

        # Customize the x-axis for the autocorrelation plot
        ax_autocorr.xaxis.set_tick_params(labelsize=12)
        ax_autocorr.yaxis.set_tick_params(labelsize=14)

        fig.align_labels()
        plt.suptitle(f"Cycle={period_combination}", fontsize=24)
        plt.subplots_adjust(top=0.95)
        output_filepath = os.path.join(self.output_dir, f"cycle_{'_'.join(map(str, period_combination))}.png")
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_heatmap(self, metrics_df):
        """
        为每个指标绘制热图。
        
        :param metrics_df: 合并后的指标 DataFrame
        """
        # 提取所有指标列，排除周期列（cycle1, cycle2）
        metric_columns = metrics_df.columns.difference(['cycle1', 'cycle2'])

        for metric in metric_columns:
            # 构建用于绘图的数据框
            data = metrics_df[['cycle1', 'cycle2', metric]].dropna()
            data.loc[data['cycle1'] == data['cycle2'], metric] = np.nan
            
            pivot_table = data.pivot_table(index="cycle2", columns="cycle1", values=metric)
            pivot_table = pivot_table.iloc[::-1]

            cmap = sns.color_palette("coolwarm", as_cmap=True)
            cmap.set_bad(color='dimgray')
            
            plt.figure(figsize=(12, 12))
            ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'shrink': .8}, annot_kws={"size": 6},
                             mask=pivot_table.isnull(), cbar=True, linewidths=.5, linecolor='lightgrey', square=True)

            plt.title(f'Heatmap for {metric}', fontsize=20, fontname='Times New Roman')
            plt.xlabel("First Cycle", fontsize=14, fontname='Times New Roman')
            plt.ylabel("Second Cycle", fontsize=14, fontname='Times New Roman')
            plt.xticks(fontsize=10, fontname='Times New Roman')
            plt.yticks(fontsize=10, fontname='Times New Roman')
            plt.tight_layout()
            
            output_filepath = os.path.join(self.output_dir, f"heatmap_{metric}.png")
            print (output_filepath)
            plt.savefig(output_filepath, dpi=600)
            # plt.show()
            plt.close()



    def plot_multiple_projections(self, selected_cycles_df, projection_results, title="Projection with Selected Cycles"):
        """
        绘制多个周期组合的预测值在同一张图表上，并计算和绘制置信区间带。
        
        :param selected_cycles_df: 包含选择的周期组合及其指标的 DataFrame。
        :param projection_results: 包含投影结果的字典，key 为周期组合，value 为对应的 DataFrame。
        :param title: 图表的标题。
        """
        plt.figure(figsize=(20, 8))  # 增大图表尺寸
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 18  # 设置整体字体大小

        # 初始化用于存储置信区间的列表
        all_projections = []
        
        # 遍历每个选定的周期组合
        for _, row in selected_cycles_df.iterrows():
            # 提取周期组合的 key
            cycles = tuple(row[[col for col in row.index if col.lower().startswith('cycle')]])

            if cycles in projection_results:
                # 获取对应的投影结果 DataFrame
                df = projection_results[cycles]

                # 将 date 列转换为 datetime 格式
                df['date'] = pd.to_datetime(df['date'])

                # 找到观测数据结束的数据点下标
                last_observation_index = df.index[~df['is_projected']].max()

                # 收集所有投影数据用于计算置信区间
                all_projections.append(df['projection'].values)

                # 绘制预测部分
                projected_dates  = df.loc[last_observation_index:, 'date']
                projected_values = df.loc[last_observation_index:, 'projection']
                plt.plot(projected_dates, projected_values, color='tab:red', lw=1.5, linestyle='--', alpha=0.4, zorder=10)

                # 绘制观测部分的拟合
                observed_dates  = df.loc[:last_observation_index, 'date']
                observed_values = df.loc[:last_observation_index, 'projection']
                plt.plot(observed_dates, observed_values, color='tab:blue', lw=1.5, alpha=0.5, zorder=11)

        # 绘制真实数据
        plt.plot(df['date'], df['observation'], color='black', lw=2.5, alpha=1.0, zorder=12)

        # 设置 x 轴刻度
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

        # 添加标签和标题
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Values', fontsize=20)
        plt.title(title, fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 添加图例
        plt.legend(fontsize=16)

        plt.tight_layout()
        plt.show()

        # 绘制置信区间带
        self.plot_confidence_interval(df['date'], all_projections, df['observation'])

    def plot_confidence_interval(self, dates, projections, observations):
        """
        绘制置信区间带，并绘制原始数据。
        
        :param dates: 日期序列
        :param projections: 多个周期组合的投影结果矩阵
        :param observations: 原始观测数据
        """
        projections = np.array(projections)

        # 计算均值和置信区间（95%）
        mean_projection = np.mean(projections, axis=0)
        lower_bound = np.percentile(projections, 2.5, axis=0)
        upper_bound = np.percentile(projections, 97.5, axis=0)

        plt.figure(figsize=(20, 8))
        
        # 绘制置信区间带
        plt.fill_between(dates, lower_bound, upper_bound, color='tab:cyan', alpha=0.2, label='95% Confidence Interval')
        plt.plot(dates, mean_projection, color='tab:blue', lw=2, label='Mean Projection')
        
        # 绘制原始数据
        plt.plot(dates, observations, color='black', lw=2.5, alpha=1.0, label='Observed Data')
        
        # 设置 x 轴刻度
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

        # 添加标签和标题
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Projection', fontsize=20)
        plt.title('Projection with Confidence Interval', fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 添加图例
        plt.legend(fontsize=16)

        plt.tight_layout()
        plt.show()