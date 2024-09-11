import pandas as pd
import numpy as np
from scipy.stats import gamma


def calculate_nll_using_fitted_gamma(df):
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


def calculate_nll_from_csv(file_path):
    """
    从 CSV 文件中读取数据并计算 NLL。

    :param file_path: CSV 文件的路径。
    :return: 计算得到的 NLL 值。
    """
    df = pd.read_csv(file_path)
    return calculate_nll_using_fitted_gamma(df)
