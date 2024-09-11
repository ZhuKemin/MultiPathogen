import pandas as pd
import numpy as np
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import gamma

class BasicMetricsCalculator:
    def __init__(self):
        pass

    def calculate_gamma_nll(self, df):
        """
        使用观测值拟合 Gamma 分布模型，并计算预测值在该分布下的 NLL。

        :param df: DataFrame，包含 'observation', 'remainder' 列。
        :return: 计算得到的 NLL 值。
        """
        observed = df['observation']
        predicted = df['observation'] - df['remainder']

        # 修正为非负数，防止 log(0) 或负数问题
        observed = np.maximum(observed, 1e-10)
        predicted = np.maximum(predicted, 1e-10)

        # 使用观测值拟合 Gamma 分布
        shape, loc, scale = gamma.fit(observed, floc=0)  # loc 固定为 0，因为 Gamma 分布常用参数形式是 a, loc, scale

        # 使用拟合的参数计算预测值的对数似然值
        log_likelihoods = gamma.logpdf(predicted, a=shape, loc=loc, scale=scale)

        # 计算 NLL
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
            nll = self.calculate_gamma_nll(df)
            acf = self.calculate_autocorr_significance(df)
            metrics_list.append({
                'period': period,
                'nll': nll,
                'acf': acf
            })
        
        # 将所有计算结果合并为一个 DataFrame
        basic_metrics = pd.DataFrame(metrics_list)

        return basic_metrics
