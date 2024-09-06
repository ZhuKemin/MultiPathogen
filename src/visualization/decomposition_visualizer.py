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
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_lag_weeks = max_lag_weeks  # 设置最大滞后周数
        plt.rcParams["font.family"] = "Times New Roman"

    def plot_decomposition_results(self, decomposition_results):
        self.global_y_limits = self._calculate_global_y_limits(decomposition_results)
        
        for period_combination, df in decomposition_results.items():
            self._plot_decomposition(period_combination, df)

    def _calculate_global_y_limits(self, decomposition_results):
        y_limits = {'trend': [np.inf, -np.inf], 'remainder': [np.inf, -np.inf], 'observed': [np.inf, -np.inf], 'seasonal': [np.inf, -np.inf]}

        for df in decomposition_results.values():
            df.columns = df.columns.str.lower()
            for key in y_limits.keys():
                if key == 'observed':
                    seasonal_columns = [col for col in df.columns if col.startswith('seasonal')]
                    observed = df['trend'] + df[seasonal_columns].sum(axis=1) + df['remainder']
                    y_limits[key][0] = min(y_limits[key][0], observed.min())
                    y_limits[key][1] = max(y_limits[key][1], observed.max())
                else:
                    columns = [col for col in df.columns if col.startswith(key)]
                    for col in columns:
                        y_limits[key][0] = min(y_limits[key][0], df[col].min())
                        y_limits[key][1] = max(y_limits[key][1], df[col].max())

        for key in y_limits.keys():
            y_range = y_limits[key][1] - y_limits[key][0]
            y_limits[key][0] -= 0.1 * y_range
            y_limits[key][1] += 0.1 * y_range

        return y_limits

    def _plot_decomposition(self, period_combination, df):
        dates = pd.to_datetime(df['date'].values)
        trend = df['trend'].values
        seasonal_columns = [col for col in df.columns if col.startswith('seasonal')]
        resid = df['remainder'].values
        observed = trend + df[seasonal_columns].sum(axis=1) + resid
        predict = trend + df[seasonal_columns].sum(axis=1)

        num_axes = 4 + len(seasonal_columns)
        fig, axs = plt.subplots(num_axes, 1, figsize=(15, 2 * num_axes), sharex=False)

        ax1 = axs[0]
        ax1.plot(dates, observed, lw=2, color='tab:blue', label='Observed')
        ax1.plot(dates, predict, lw=1, ls='--', color='tab:red', label='Predicted')
        ax1.set_ylabel('Observed', fontsize=18)
        ax1.set_ylim(self.global_y_limits['observed'])

        ax_trend = axs[1]
        ax_trend.plot(dates, trend, lw=2, color='tab:green')
        ax_trend.set_ylabel('Trend', fontsize=18)
        ax_trend.set_ylim(self.global_y_limits['trend'])

        for i, col in enumerate(seasonal_columns):
            ax_seasonal = axs[2 + i]
            cycle_value = int(round(float(col.replace('seasonal', '')))) if col!='seasonal' else 12
            ax_seasonal.plot(dates, df[col].values, lw=2, color='tab:orange')
            ax_seasonal.set_ylabel(f'Seasonal {cycle_value}', fontsize=18)
            ax_seasonal.set_ylim(self.global_y_limits['seasonal'])
            start_date = dates[0]
            while start_date <= dates[-1]:
                ax_seasonal.axvline(x=start_date, color='darkgray', linestyle='--')
                start_date += pd.DateOffset(weeks=cycle_value)

        ax_resid = axs[2 + len(seasonal_columns)]
        ax_resid.scatter(dates, resid, color='tab:red')
        ax_resid.set_ylabel('Residual', fontsize=18)
        ax_resid.set_ylim(self.global_y_limits['remainder'])

        correlations = ccf(resid, resid, adjusted=True)
        lags = np.arange(-self.max_lag_weeks, self.max_lag_weeks + 1)
        correlations_combined = np.concatenate((correlations[1:self.max_lag_weeks + 1][::-1], correlations[:self.max_lag_weeks + 1]))

        ax_autocorr = axs[-1]
        ax_autocorr.stem(lags, correlations_combined, linefmt='-', markerfmt='o', basefmt='r-')
        ax_autocorr.grid(which='both', linestyle='-.', linewidth=0.5, color='lightgrey')
        ax_autocorr.set_ylabel('Correlation', fontsize=18)
        ax_autocorr.set_xlim(-self.max_lag_weeks, self.max_lag_weeks)

        for ax in axs[:-1]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.grid(which='both', linestyle='-.', linewidth=0.4, color='lightgrey')
            ax.set_axisbelow(True)
            ax.set_xlim(dates[0], dates[-1])

        ax_autocorr.xaxis.set_tick_params(labelsize=12)
        ax_autocorr.yaxis.set_tick_params(labelsize=14)

        fig.align_labels()
        plt.suptitle(f"Cycle={period_combination}", fontsize=24)
        plt.subplots_adjust(top=0.95)
        output_filepath = os.path.join(self.output_dir, f"decomposition_cycle_{'_'.join(map(str, period_combination))}.png")
        plt.tight_layout()
        # plt.show()
        plt.savefig(output_filepath, dpi=300)
        plt.close()

    def plot_heatmap(self, metrics_df, frequency):
        metric_columns = metrics_df.columns.difference(['cycle1', 'cycle2'])
        
        # 决定高亮行是 12 还是 52
        highlight_row = 12 if frequency == 'monthly' else 52
        
        for metric in metric_columns:
            data = metrics_df[['cycle1', 'cycle2', metric]].dropna()
            data.loc[data['cycle1'] == data['cycle2'], metric] = np.nan
            
            pivot_table = data.pivot_table(index="cycle2", columns="cycle1", values=metric)
            pivot_table = pivot_table.iloc[::-1]
            
            cmap = sns.color_palette("coolwarm_r", as_cmap=True)
            cmap.set_bad(color='dimgray')
            
            plt.figure(figsize=(12, 12))
            ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'shrink': .78}, annot_kws={"size": 8},
                             mask=pivot_table.isnull(), cbar=True, linewidths=.5, linecolor='lightgrey', square=True)

            # 高亮指定的行
            row_to_highlight = pivot_table.index.get_loc(highlight_row)
            for i in range(pivot_table.shape[1]):
                ax.add_patch(plt.Rectangle((i, row_to_highlight), 1, 1, fill=False, edgecolor='dimgray', lw=0.5))

            # 其余绘图逻辑
            plt.title(f'Heatmap for {metric}', fontsize=20, fontname='Times New Roman')
            plt.xlabel("First Cycle", fontsize=14, fontname='Times New Roman')
            plt.ylabel("Second Cycle", fontsize=14, fontname='Times New Roman')
            plt.tight_layout()
            
            output_filepath = os.path.join(self.output_dir, f"heatmap_{metric}.png")
            plt.show()
            # plt.savefig(output_filepath, dpi=600)
            plt.close()
            exit()



    def plot_heatmap_bak(self, metrics_df):
        metric_columns = metrics_df.columns.difference(['cycle1', 'cycle2'])

        for metric in metric_columns:
            data = metrics_df[['cycle1', 'cycle2', metric]].dropna()
            data.loc[data['cycle1'] == data['cycle2'], metric] = np.nan
            
            pivot_table = data.pivot_table(index="cycle2", columns="cycle1", values=metric)
            pivot_table = pivot_table.iloc[::-1]

            # 倒转冷暖色调，值越小越红
            vmin,vmax = metrics_df[metric].min(), metrics_df[metric].max()
            cmap = sns.color_palette("coolwarm_r", as_cmap=True)
            cmap2 = cmap.copy()
            cmap2.set_bad(color='black')

            # 计算前90%的阈值
            threshold_value = pivot_table.quantile(0.1).max()
            mask = pivot_table > threshold_value

            plt.figure(figsize=(12, 12))
            ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'shrink': .8}, annot_kws={"size": 8},
                             mask=mask, cbar=True, linewidths=.5, linecolor='lightgrey', square=True, alpha=1.0, vmin=vmin, vmax=vmax)
            
            # 将大于阈值的网格透明度调低
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap2, annot_kws={"size": 8, "color": "gray"},
                             mask=~mask, cbar=False, linewidths=.5, linecolor='lightgrey', square=True, alpha=0.15, vmin=vmin, vmax=vmax, ax=ax)

            # 高亮12那一行的最佳取值
            row_to_highlight = pivot_table.index.get_loc(12)
            best_in_row_cycle = pivot_table.loc[12].idxmin()
            best_in_row_index = pivot_table.columns.get_loc(best_in_row_cycle)
            
            ax.add_patch(plt.Rectangle((best_in_row_index, row_to_highlight), 1, 1, fill=False, edgecolor='black', lw=2))

            # 高亮全局最小值
            min_value = pivot_table.min().min()
            min_positions = np.where(pivot_table == min_value)

            for y, x in zip(min_positions[0], min_positions[1]):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='white', lw=2))
            
            # 标出12行
            for i in range(pivot_table.shape[1]):
                ax.add_patch(plt.Rectangle((i, row_to_highlight), 1, 1, fill=False, edgecolor='dimgray', lw=0.5))

            plt.title(f'Heatmap for {metric}', fontsize=20, fontname='Times New Roman')
            plt.xlabel("First Cycle", fontsize=14, fontname='Times New Roman')
            plt.ylabel("Second Cycle", fontsize=14, fontname='Times New Roman')
            plt.xticks(fontsize=10, fontname='Times New Roman')
            plt.yticks(fontsize=10, fontname='Times New Roman')
            plt.tight_layout()
            
            output_filepath = os.path.join(self.output_dir, f"heatmap_{metric}.png")
            # plt.savefig(output_filepath, dpi=600)
            plt.show()
            plt.close()

    def plot_multiple_projections(self, selected_cycles_df, projection_results, title="Projection with Selected Cycles"):
        plt.figure(figsize=(20, 8))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 18

        all_projections = []

        for _, row in selected_cycles_df.iterrows():
            cycles = tuple(row[[col for col in row.index if col.lower().startswith('cycle')]])

            if cycles in projection_results:
                df = projection_results[cycles]
                df['date'] = pd.to_datetime(df['date'])
                last_observation_index = df.index[~df['is_projected']].max()

                all_projections.append(df['projection'].values)

                projected_dates  = df.loc[last_observation_index:, 'date']
                projected_values = df.loc[last_observation_index:, 'projection']
                plt.plot(projected_dates, projected_values, color='tab:red', lw=0.5, linestyle='--', alpha=0.3, zorder=10)

                observed_dates  = df.loc[:last_observation_index, 'date']
                observed_values = df.loc[:last_observation_index, 'projection']
                plt.plot(observed_dates, observed_values, color='tab:blue', lw=0.5, alpha=0.4, zorder=11)

        # TODO: 处理分解模型中两个周期相等的特殊情况
        df =[v for k,v in projection_results.items() if k[0]!=k[1]][0]
        plt.plot(df['date'], df['observation'], color='black', lw=2.0, alpha=1.0, zorder=12)

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 设置 x 轴日期格式为 年-月

        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Values', fontsize=20)
        plt.title(title, fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.ylim(bottom=0)

        output_filepath = os.path.join(self.output_dir, "projections_selected_cycles.png")
        # plt.savefig(output_filepath, dpi=600)
        plt.show()
        plt.close()

        # self.plot_confidence_interval(df['date'], all_projections, df['observation'])

    def plot_confidence_interval(self, dates, projections, observations):
        projections = np.array(projections)

        mean_projection = np.mean(projections, axis=0)
        lower_bound = np.percentile(projections, 2.5, axis=0)
        upper_bound = np.percentile(projections, 97.5, axis=0)

        plt.figure(figsize=(20, 8))
        plt.fill_between(dates, lower_bound, upper_bound, color='tab:cyan', alpha=0.2, label='95% Confidence Interval')
        plt.plot(dates, mean_projection, color='tab:blue', lw=2, label='Mean Projection')
        plt.plot(dates, observations, color='black', lw=2.5, alpha=1.0, label='Observed Data')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Projection', fontsize=20)
        plt.title('Projection with Confidence Interval', fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=16)
        plt.tight_layout()

        output_filepath = os.path.join(self.output_dir, "projections_confidence_interval.png")
        plt.savefig(output_filepath, dpi=600)
        plt.close()


    def plot_combined_top_projections(self, city, base_dir, top_combinations_df, title="Combined Top Projections for RSV and Flu"):
        """
        绘制选定组合的叠加投影图和观测数据。
        :param city: 城市名称。
        :param base_dir: 数据所在的基础路径。
        :param top_combinations_df: 包含选定组合的 DataFrame。
        :param title: 图表标题。
        """
        plt.figure(figsize=(20, 6))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 18

        # 初始化存储投影数据的字典
        projections_flu = {}
        projections_rsv = {}

        # 分别加载 RSV 和 Flu 的所有投影数据
        flu_dir = os.path.join(base_dir, f"{city}-flu", "cross_year", "projection")
        rsv_dir = os.path.join(base_dir, f"{city}-rsv", "cross_year", "projection")

        for file_name in os.listdir(flu_dir):
            if file_name.endswith('.csv'):
                cycle_flu = file_name.replace('cycle_', '').replace('.csv', '')
                df_flu = pd.read_csv(os.path.join(flu_dir, file_name))
                df_flu['date'] = pd.to_datetime(df_flu['date'])

                # 按月份进行汇总计算平均值
                df_flu['month'] = df_flu['date'].dt.to_period('M').dt.to_timestamp()
                df_flu_grouped = df_flu.groupby('month').agg({'projection': 'mean', 'observation': 'mean'}).reset_index()
                projections_flu[cycle_flu] = df_flu_grouped

        for file_name in os.listdir(rsv_dir):
            if file_name.endswith('.csv'):
                cycle_rsv = file_name.replace('cycle_', '').replace('.csv', '')
                df_rsv = pd.read_csv(os.path.join(rsv_dir, file_name))
                df_rsv['date'] = pd.to_datetime(df_rsv['date'])

                # 按月份进行汇总计算平均值
                df_rsv['month'] = df_rsv['date'].dt.to_period('M').dt.to_timestamp()
                df_rsv_grouped = df_rsv.groupby('month').agg({'projection': 'mean', 'observation': 'mean'}).reset_index()
                projections_rsv[cycle_rsv] = df_rsv_grouped

        # 计算所有病种中 projection 列的全局最大值
        max_projection_value = max(
            max(df['projection'].max() for df in projections_flu.values()),
            max(df['projection'].max() for df in projections_rsv.values())
        )

        # 遍历 top_combinations_df 并绘制每个组合的图
        for _, row in top_combinations_df.iterrows():
            cycle_flu = row['cycle_flu']
            cycle_rsv = row['cycle_rsv']

            # 获取对应周期组合的 RSV 和 Flu 数据
            df_flu_grouped = projections_flu.get(cycle_flu)
            df_rsv_grouped = projections_rsv.get(cycle_rsv)

            if df_flu_grouped is not None and df_rsv_grouped is not None:
                # 数据标准化到新列 'normalized_projection'
                df_flu_grouped['normalized_projection'] = df_flu_grouped['projection'] / max_projection_value
                df_rsv_grouped['normalized_projection'] = df_rsv_grouped['projection'] / max_projection_value

                # 合并时间范围为交集，并进行累加
                current_combined_dates = pd.concat([df_flu_grouped['month'], df_rsv_grouped['month']]).drop_duplicates().sort_values()
                current_combined_dates = current_combined_dates[
                    (current_combined_dates >= max(df_flu_grouped['month'].min(), df_rsv_grouped['month'].min())) & 
                    (current_combined_dates <= min(df_flu_grouped['month'].max(), df_rsv_grouped['month'].max()))
                ]

                current_projections = []
                for date in current_combined_dates:
                    flu_proj = df_flu_grouped[df_flu_grouped['month'] == date]['normalized_projection'].sum() if not df_flu_grouped[df_flu_grouped['month'] == date].empty else 0
                    rsv_proj = df_rsv_grouped[df_rsv_grouped['month'] == date]['normalized_projection'].sum() if not df_rsv_grouped[df_rsv_grouped['month'] == date].empty else 0
                    current_projections.append(flu_proj + rsv_proj)

                # 绘制叠加的投影图
                plt.plot(current_combined_dates, current_projections, color='tab:blue', lw=0.5, linestyle='--', alpha=0.3)

        # # 绘制观测数据
        df_flu_obs = df_flu_grouped.loc[df_flu_grouped.month.isin(current_combined_dates), 'observation'].values/max_projection_value
        df_rsv_obs = df_rsv_grouped.loc[df_rsv_grouped.month.isin(current_combined_dates), 'observation'].values/max_projection_value
        plt.plot(current_combined_dates, df_flu_obs+df_rsv_obs, color='black', lw=2.0, label="Observation" )

        # 设置 x 轴为日期格式，并自动格式化刻度标签
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # 每2年显示一个主刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化为 'YYYY-MM'
        plt.xticks(rotation=0)  # 标签旋转角度

        # 目前的数据最早从2015开始
        ax.set_xlim(pd.Timestamp('2005-01-01'), pd.Timestamp('2035-12-31'))
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('HealthCare Burden', fontsize=20)
        plt.title(title, fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        output_filepath = os.path.join(base_dir, f"{city}_combined_top_projections.png")
        plt.savefig(output_filepath, dpi=600)
        # plt.show()
        plt.close()



if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import itertools

    DV = DecompositionVisualizer(r'E:/MultiPathogen/results/Suzhou-flu/cross_year/metrics')

    # 生成12到40之间的所有整数组合
    cycles = range(12, 41)
    combinations = list(itertools.product(cycles, repeat=2))

    # 生成0到1之间的随机小数
    acf_values = np.random.rand(len(combinations))

    # 构建DataFrame
    df = pd.DataFrame(combinations, columns=['cycle1', 'cycle2'])
    df['acf'] = acf_values

    DV.plot_heatmap(df)