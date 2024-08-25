import pandas as pd

class DerivedMetricsCalculator:
    def __init__(self):
        pass

    def calculate_avg_rank(self, basic_metrics):
        """
        计算基础指标排名的平均值。
        
        :param basic_metrics: 包含多个基础指标的DataFrame，列名为指标名称
        :return: 每个周期组合的排名平均值 (Series)
        """
        # 分别对 nll 和 acf 进行排名
        nll_rank = basic_metrics['nll'].rank(ascending=True)
        acf_rank = basic_metrics['acf'].rank(ascending=True)
        
        # 计算每个周期组合的排名平均值
        avg_rank = (nll_rank + acf_rank) / 2
        
        return avg_rank

    def calculate_norm_prod(self, basic_metrics):
        """
        计算基础指标归一化后的乘积。
        
        :param basic_metrics: 包含多个基础指标的DataFrame，列名为指标名称
        :return: 每个周期组合的标准化乘积 (Series)
        """
        # 对每个指标单独进行标准化
        normalized_nll = (basic_metrics['nll'] - basic_metrics['nll'].min()) / (basic_metrics['nll'].max() - basic_metrics['nll'].min())
        normalized_acf = (basic_metrics['acf'] - basic_metrics['acf'].min()) / (basic_metrics['acf'].max() - basic_metrics['acf'].min())
        
        # 计算每个周期组合的标准化值乘积
        norm_prod = normalized_nll * normalized_acf
        
        return norm_prod

    def calculate_derive_metrics(self, basic_metrics):
        """
        计算所有衍生指标。
        
        :param basic_metrics: 包含基础指标的DataFrame，列名为基础指标名称
        :return: 包含衍生指标的DataFrame
        """
        avg_rank = self.calculate_avg_rank(basic_metrics)
        norm_prod = self.calculate_norm_prod(basic_metrics)

        derived_metrics = pd.DataFrame({
            'avg_rank': avg_rank,
            'norm_prod': norm_prod
        })

        return derived_metrics