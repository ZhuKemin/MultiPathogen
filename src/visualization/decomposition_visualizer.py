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
        plt.savefig(output_filepath, dpi=300)
        plt.close()

    def plot_heatmap(self, metrics_df):
        metric_columns = metrics_df.columns.difference(['cycle1', 'cycle2'])

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

            # 高亮全局最小值
            min_value = pivot_table.min().min()
            min_positions = np.where(pivot_table == min_value)

            for y, x in zip(min_positions[0], min_positions[1]):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='white', lw=3.0))
            
            # 标出12行
            row_to_highlight = pivot_table.index.get_loc(12)
            for i in range(pivot_table.shape[1]):
                ax.add_patch(plt.Rectangle((i, row_to_highlight), 1, 1, fill=False, edgecolor='dimgray', lw=0.5))

            # 高亮12那一行的最佳取值
            best_in_row_cycle = pivot_table.loc[12].idxmin()
            best_in_row_index = pivot_table.columns.get_loc(best_in_row_cycle)
            ax.add_patch(plt.Rectangle((best_in_row_index, row_to_highlight), 1, 1, fill=False, edgecolor='black', lw=2.5))

            plt.title(f'Heatmap for {metric}', fontsize=20, fontname='Times New Roman')
            plt.xlabel("First Cycle", fontsize=14, fontname='Times New Roman')
            plt.ylabel("Second Cycle", fontsize=14, fontname='Times New Roman')
            plt.xticks(fontsize=10, fontname='Times New Roman')
            plt.yticks(fontsize=10, fontname='Times New Roman')
            plt.tight_layout()
                
            output_filepath = os.path.join(self.output_dir, f"heatmap_{metric}.png")
            # plt.show()
            plt.savefig(output_filepath, dpi=600)
            plt.close()



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

        plt.plot(df['date'], df['observation'], color='black', lw=2.0, alpha=1.0, zorder=12)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Values', fontsize=20)
        plt.title(title, fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=16)
        plt.tight_layout()

        output_filepath = os.path.join(self.output_dir, "projections_selected_cycles.png")
        plt.savefig(output_filepath, dpi=600)
        plt.close()

        self.plot_confidence_interval(df['date'], all_projections, df['observation'])

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