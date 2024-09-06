import numpy as np
import pandas as pd

class ProjectionModel:
    def __init__(self, end_date, frequency='monthly'):
        """
        :param end_date: 预测的终止日期
        :param frequency: 数据频率，'monthly' 或 'weekly'
        """
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency

    def project(self, df, cycles):
        """
        根据分解后的 DataFrame 生成未来的预测序列，并返回更新后的 DataFrame。
        
        :param df: 分解后的时间序列 DataFrame，包含 'date', 'trend', 'remainder' 和若干 'seasonal' 列。
        :param cycles: tuple，包含用于投影的周期组合 (如 (12,) 或 (12, 21))
        :return: 包含预测值的新列 'projection' 和 'is_projected' 标签的更新后的 DataFrame。
        """
        # 确保所有列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 获取现有的最后一个日期
        last_date = pd.to_datetime(df['date'].values[-1])
        
        # 根据频率生成未来的日期
        if self.frequency == 'monthly':
            # 对于月度数据，根据最后一个观测日期，生成下个月的同一天
            future_dates = pd.date_range(last_date + pd.DateOffset(months=1), self.end_date, freq=pd.DateOffset(months=1))
        elif self.frequency == 'weekly':
            # 对于周度数据，严格按照最后一个观测日期，生成下一周的日期
            future_dates = pd.date_range(last_date + pd.DateOffset(weeks=1), self.end_date, freq=pd.DateOffset(weeks=1))
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")
        
        future_dates = future_dates.strftime('%Y-%m-%d')  # 转换日期格式为与原数据一致的格式

        # 计算实际的 future steps
        future_steps = len(future_dates)

        if future_steps == 0:
            return df  # 如果没有未来的日期需要预测，返回原始 DataFrame

        # 扩展 DataFrame，添加未来的日期
        future_df = pd.DataFrame({
            'date': future_dates,
            'is_projected': True  # 标记为预测数据
        })
        
        # 扩展趋势分量
        last_trend_value = df['trend'].values[-1]
        future_df['trend'] = np.full(future_steps, last_trend_value)

        # 扩展周期性分量，并生成新列
        for cycle in cycles:
            seasonal_col = f'seasonal{cycle}'.lower()
            if seasonal_col in df.columns:
                cycle_length = cycle
                repeats = (future_steps + cycle_length - 1) // cycle_length
                extended_seasonal = np.tile(df[seasonal_col].values, repeats)[:future_steps]
                future_df[seasonal_col] = extended_seasonal

        # 合并扩展后的 DataFrame
        df['is_projected'] = False  # 标记为观测数据
        combined_df = pd.concat([df, future_df], ignore_index=True)

        # 生成投影列：趋势分量和所有季节性分量之和
        seasonal_columns = [f'seasonal{cycle}'.lower() for cycle in cycles] if 'seasonal' not in df.columns else ['seasonal']
        combined_df['projection'] = combined_df['trend'] + combined_df[seasonal_columns].sum(axis=1)

        # 将投影列中的负值设为 0
        combined_df['projection'] = combined_df['projection'].clip(lower=0)

        return combined_df
