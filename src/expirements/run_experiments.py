import sys
import os

# 获取当前文件的路径，并设置项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)

import yaml
import pandas as pd
from src.models.decomposition_model import DecompositionModel
from src.data.load_data import load_data, load_metadata
from src.evaluation.basic_metrics import BasicMetricsCalculator
from src.evaluation.derived_metrics import DerivedMetricsCalculator

class ExperimentRunner:
    def __init__(self, pathogen, city=None, filename=None, scenarios=None, parallel=True, visualize=False, config_path="src/expirements/config.yaml", metadata_path="data/metadata.yaml"):
        """
        初始化实验运行器
        :param pathogen: 病原体类型，如 "flu" 或 "rsv"
        :param city: 城市名称，如 "Beijing"
        :param filename: 数据文件名，如 "Chen-2023-Xian-Cases.csv"
        :param scenarios: 要计算的场景列表
        :param parallel: 是否启用并行计算，默认为True
        :param visualize: 是否生成可视化结果
        :param config_path: 配置文件路径，默认为 'src/expirements/config.yaml'
        :param metadata_path: 元数据文件路径，默认为 'data/metadata.yaml'
        """
        self.pathogen = pathogen
        self.city = city
        self.filename = filename
        self.scenarios = scenarios or []
        self.parallel = parallel
        self.visualize = visualize
        self.config_path = config_path
        self.metadata_path = metadata_path
        
        # 读取配置文件
        self.config = self._load_config()

        # 加载元数据
        self.metadata = self._load_metadata()

        # 加载数据集
        self.dataset = self._get_dataset()

    def _load_config(self):
        """
        从配置文件中加载详细的实验配置
        """
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def _load_metadata(self):
        """
        加载元数据文件以获取数据集的时间频率等信息
        """
        return load_metadata(self.metadata_path)

    def _get_dataset(self):
        """
        根据病原体类型、城市或文件名加载数据集，并返回相应的时间序列数据
        """
        base_path = self.config['data']['base_paths'][self.pathogen]  # 从config.yaml中读取基础路径
        
        if self.filename:
            # 当提供了文件名时，直接加载
            return load_data(file_path=self.filename)
        elif self.city:
            # 根据城市名加载数据
            return load_data(city=self.city, base_path=base_path)
        else:
            raise ValueError("必须提供文件名或城市名称中的一个来加载数据集。")

    def run(self):
        """
        执行实验，根据场景和配置文件中的设定
        """
        basic_calculator = BasicMetricsCalculator()
        derived_calculator = DerivedMetricsCalculator()

        # 初始化 DecompositionModel
        model = DecompositionModel()

        # 遍历每个场景
        for scenario in self.scenarios:
            print(f"Running scenario {scenario}")
            scenario_results = []

            periods = self.config['scenarios'][scenario]['periods'][self.metadata.get('frequency', 'monthly')]

            # 运行分解模型
            decomposition_result = model.decompose(self.dataset, periods)

            # 保存分解结果
            self._save_decomposition_result(decomposition_result, scenario, periods)

            # 计算基础指标
            basic_metrics = basic_calculator.calculate_basic_metrics(decomposition_result)

            # 计算衍生指标
            derived_metrics = derived_calculator.calculate_derive_metrics(basic_metrics)

            # 合并基础指标和衍生指标
            combined_metrics = pd.concat([basic_metrics, derived_metrics], axis=1)

            # 将周期信息添加到结果中
            combined_metrics['periods'] = str(periods)
            scenario_results.append(combined_metrics)

            # 将场景结果保存到文件
            self._save_metrics(scenario, scenario_results)

    def _save_decomposition_result(self, decomposition_result, scenario, periods):
        """
        保存时间序列分解结果，每个周期组合保存为单独的文件。
        
        :param decomposition_result: 包含不同周期组合分解结果的字典
        :param scenario: 当前运行的场景名称
        :param periods: 当前场景下的周期组合列表
        """
        output_dir = os.path.join(self.config['output']['result_dir'], self.pathogen, self.city, scenario)
        os.makedirs(output_dir, exist_ok=True)

        # 遍历字典，分别保存每个周期组合的分解结果
        for period, df in decomposition_result.items():
            period_str = '_'.join(map(str, period))
            output_path = os.path.join(output_dir, f'cycle_{period_str}.csv')
            df.to_csv(output_path, index=False)
            print(f"Decomposition result saved to {output_path}")

    def _save_metrics(self, scenario, scenario_results):
        """
        将场景下的指标（包括基础和衍生指标）保存到一个表中。
        
        :param scenario: 当前运行的场景名称
        :param scenario_results: 当前场景的结果列表
        """
        output_dir = os.path.join(self.config['output']['result_dir'], self.pathogen, self.city, scenario)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics.csv')

        final_df = pd.concat(scenario_results, ignore_index=True)

        # 拆解 period 列为 cycle1, cycle2, ...
        if 'period' in final_df.columns:
            period_columns = final_df['period'].apply(pd.Series)
            period_columns.columns = [f'cycle{i+1}' for i in range(period_columns.shape[1])]
            final_df = pd.concat([period_columns, final_df.drop(columns=['periods'])], axis=1)

        final_df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")


    def _generate_visualization(self, result, scenario):
        """
        生成并保存结果的可视化图表
        """
        # 可视化代码，生成并保存图表
        pass


if __name__ == "__main__":
    import os

    # 设置工作目录为项目的根目录
    os.chdir('E:/MultiPathogen')
    print(f"Current working directory: {os.getcwd()}")

    runner = ExperimentRunner(
        pathogen="flu",
        city="Beijing",
        # scenarios=["baseline", "cross_year", "complex"],
        # scenarios=["baseline", "cross_year"],
        scenarios=["cross_year"],
        visualize=True
    )
    runner.run()
