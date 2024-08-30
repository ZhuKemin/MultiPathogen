import numpy as np
import pandas as pd

class ProjectionModel:
    def __init__(self, future_steps=12):
        self.future_steps = future_steps

    def project(self, df, cycles):
        """
        根据分解后的 DataFrame 生成未来的预测序列，并返回更新后的 DataFrame。
        
        :param df: 分解后的时间序列 DataFrame，包含 'date', 'trend', 'remainder' 和若干 'seasonal' 列。
        :param cycles: tuple，包含用于投影的周期组合 (如 (12,) 或 (12, 21))
        :return: 包含预测值的新列 'projection' 和 'is_projected' 标签的更新后的 DataFrame。
        """
        # 确保所有列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 获取现有的日期
        last_date = pd.to_datetime(df['date'].values[-1])
        
        # 创建扩展的日期，确保格式与原始日期格式一致
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=self.future_steps, freq='ME')
        future_dates = future_dates.strftime('%Y-%m-%d')  # 转换日期格式为与原数据一致的格式

        # 扩展 DataFrame，添加未来的日期
        future_df = pd.DataFrame({
            'date': future_dates,
            'is_projected': True  # 标记为预测数据
        })
        
        # 扩展趋势分量
        last_trend_value = df['trend'].values[-1]
        future_df['trend'] = np.full(self.future_steps, last_trend_value)

        # 扩展周期性分量，并生成新列
        for cycle in cycles:
            seasonal_col = f'seasonal{cycle}'.lower()
            if seasonal_col in df.columns:
                cycle_length = cycle  # 直接使用 key 中的周期长度
                repeats = (self.future_steps + cycle_length - 1) // cycle_length
                extended_seasonal = np.tile(df[seasonal_col].values, repeats)[:self.future_steps]
                future_df[seasonal_col] = extended_seasonal

        # 合并扩展后的 DataFrame
        df['is_projected'] = False  # 标记为观测数据
        combined_df = pd.concat([df, future_df], ignore_index=True)

        # 生成投影列：趋势分量和所有季节性分量之和
        seasonal_columns = [f'seasonal{cycle}'.lower() for cycle in cycles] if 'seasonal' not in df.columns else ['seasonal']
        combined_df['projection'] = combined_df['trend'] + combined_df[seasonal_columns].sum(axis=1)

        # 将投影列中的负值设为 0
        combined_df['projection'] = combined_df['projection'].clip(lower=0)

        # 生成观测列：对观测部分为原始数据（趋势分量、所有季节性分量与残差之和），对预测部分为 nan
        combined_df['observation'] = combined_df['projection'] + combined_df['remainder']

        return combined_df
