import pandas as pd
import numpy as np
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox

class BasicMetricsCalculator:
    def __init__(self):
        pass

    def calculate_poisson_nll(self, df):
        """
        计算泊松分布的负极大似然率（NLL）。
        
        :param df: 时间序列分解后的DataFrame，包含'Trend', 'Seasonal' 列 和 'remainder' 列
        :return: 负极大似然率值
        """
        # 计算预测值：trend + 所有 Seasonal 列的和
        predicted = df['trend'] + df[[col for col in df.columns if 'Seasonal' in col]].sum(axis=1)
        
        # 确保所有预测值均大于零
        predicted = np.maximum(predicted, 1e-10)
            
        # 实际值应为预测值 + 残差
        observed = predicted + df['remainder']
        
        # 泊松分布的对数似然计算
        log_likelihoods = observed * np.log(predicted) - predicted - gammaln(observed + 1)
        
        # 计算负极大似然率
        nll = -np.sum(log_likelihoods)
        
        return nll

    def calculate_autocorr_significance(self, df, lag=10):
        """
        计算残差序列的指定滞后期的Ljung-Box显著性p值。
        
        :param df: 时间序列分解后的DataFrame，包含'Remainder'列
        :param lag: 计算显著性的滞后期，默认值为10
        :return: 滞后期为lag的Ljung-Box检验p值
        """
        remainder = df['remainder'].values
        ljung_box_result = acorr_ljungbox(remainder, lags=[lag], return_df=True)
        p_value = ljung_box_result['lb_pvalue'].iloc[0]
        return -p_value

    def calculate_basic_metrics(self, decomposition_results):
        """
        计算所有基础指标。
        
        :param decomposition_results: 包含时间序列分解结果的字典，其中每个键对应一个周期组合，值为DataFrame
        :return: 包含基础指标的DataFrame
        """
        metrics_list = []

        for period, df in decomposition_results.items():
            # 计算基础指标
            nll = self.calculate_poisson_nll(df)
            acf = self.calculate_autocorr_significance(df)
            metrics_list.append({
                'period': period,
                'nll': nll,
                'acf': acf
            })
        
        # 将所有计算结果合并为一个 DataFrame
        basic_metrics = pd.DataFrame(metrics_list)

        return basic_metrics
