import sys
import os

# 获取当前文件的路径，并设置项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)


import yaml
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.decomposition_model import DecompositionModel
from src.data.load_data import load_data, load_metadata
from src.evaluation.basic_metrics import BasicMetricsCalculator
from src.evaluation.derived_metrics import DerivedMetricsCalculator
from src.visualization.decomposition_visualizer import DecompositionVisualizer
from src.projection.projection_model import ProjectionModel

# TODO: 增加对周期组合超过半数据长度的判断

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

        # 提取频率信息
        self.frequency = self.metadata.get('proc_data_freq') 

        # 加载数据集
        self.dataset = self._get_dataset()

        # 构造输出目录路径
        self.output_dir_base = os.path.join(self.config['output']['result_dir'], f"{self.city}-{self.pathogen}")


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
        return load_metadata(self.metadata_path)[self.city][self.pathogen]

    def _get_dataset(self):
        """
        根据病原体类型、城市或文件名加载数据集，并返回相应的时间序列数据
        """
        base_path = self.config['data']['base_paths'][self.pathogen]  # 从config.yaml中读取基础路径

        # 使用load_data方法加载数据
        return load_data(city=self.city, file_path=self.filename, base_path=base_path, pathogen=self.pathogen, metadata_path=self.metadata_path)

    def _process_combination(self, period_combination, model):
        """
        处理单个周期组合的分解任务。

        :param period_combination: 周期组合，可能是一个列表或元组，表示要分析的周期长度
        :param model: DecompositionModel 实例，用于执行时间序列分解
        :return: 返回周期组合和分解结果 (DataFrame)
        """
        print(f"Processing period combination: {period_combination}")
        decomposition_result = model.decompose(self.dataset, period_combination)
        return period_combination, decomposition_result

    # TODO: 复杂场景的逻辑
    def _execute_combinations(self, periods, model, scenario_results):
        """
        执行所有周期组合的分解任务，根据配置选择并行或顺序执行。

        :param periods: 场景中的周期设定，可以是列表或字典
        :param model: DecompositionModel 实例，用于执行时间序列分解
        :param scenario_results: 存储当前场景的分解结果列表
        :return: 所有周期组合的分解结果
        """
        data_length = len(self.dataset)

        if isinstance(periods, list):
            combinations = [tuple(periods)]  # 转换为单一组合的元组形式
        elif isinstance(periods, dict):
            # 支持两个 range 的 cross_year 场景
            if "range1" in periods and "range2" in periods:
                # 限制 range1 的最大值为数据长度的一半
                max_range1_value = min(periods['range1'][1], (data_length-1) // 2)
                range1 = range(periods['range1'][0], max_range1_value + 1)

                # 限制 range2 的最大值为数据长度的一半
                max_range2_value = min(periods['range2'][1], (data_length-1) // 2)
                range2 = range(periods['range2'][0], max_range2_value + 1)
                
                combinations = [(r1, r2) for r1, r2 in product(range1, range2) if r1!=r2]
            else:
                raise ValueError("For cross_year scenario, both 'range1' and 'range2' must be provided.")
        else:
            raise ValueError("Unsupported periods type. Must be either list or dict.")

        decomposition_results = {}

        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._process_combination, comb, model) for comb in combinations]
                for future in as_completed(futures):
                    period_combination, decomposition_result = future.result()
                    decomposition_results[period_combination] = decomposition_result
        else:
            for comb in combinations:
                period_combination, decomposition_result = self._process_combination(comb, model)
                decomposition_results[period_combination] = decomposition_result

        return decomposition_results

    def _calculate_metrics(self, decomposition_results, basic_calculator, derived_calculator):
        """
        计算所有周期组合的基础指标和衍生指标，并合并为最终的指标 DataFrame。

        :param decomposition_results: 所有分解结果的字典，键为周期组合，值为DataFrame
        :param basic_calculator: BasicMetricsCalculator 实例，用于计算基础指标
        :param derived_calculator: DerivedMetricsCalculator 实例，用于计算衍生指标
        :return: 合并后的基础指标和衍生指标 (final_df)
        """
        basic_metrics = basic_calculator.calculate_basic_metrics(decomposition_results)
        derived_metrics = derived_calculator.calculate_derive_metrics(basic_metrics)
        
        # 合并基础指标和衍生指标
        final_df = pd.concat([basic_metrics, derived_metrics], axis=1)

        # 拆解 period 列为 cycle1, cycle2, ...
        if 'period' in final_df.columns:
            period_columns = final_df['period'].apply(pd.Series)
            period_columns.columns = [f'cycle{i+1}' for i in range(period_columns.shape[1])]
            final_df = pd.concat([period_columns, final_df], axis=1).drop(columns=['period'])

        return final_df

    def _process_projections(self, decomposition_results, projection_model):
        """
        对每个分解结果进行投影处理，生成带有投影数据的结果字典。

        :param decomposition_results: 分解结果的字典，key 为周期组合，value 为分解后的 DataFrame。
        :param projection_model: ProjectionModel 实例，用于生成投影数据。
        :return: 包含投影数据的新字典，key 与 decomposition_results 相同，value 为带有投影数据的 DataFrame。
        """
        def apply_projection(key, df):
            return projection_model.project(df, cycles=key)
        
        # 使用 map 函数将投影应用于每个分解结果
        projection_results = {key: apply_projection(key, df) for key, df in decomposition_results.items()}
        
        return projection_results

    def select_top_cycles(self, metrics_df, metric, percent, ascending=True):
        """
        根据指定的指标和百分比，选取前百分之几的周期组合。
        
        :param metrics_df: 包含周期组合及其指标得分的 DataFrame
        :param metric: 用于筛选的具体指标名称（如 'nll'）
        :param percent: 选择的百分比（如 5 表示前 5%）
        :param ascending: 如果为 True，选择最小的指标值；如果为 False，选择最大的指标值。
        :return: 包含选定周期组合及其得分的 DataFrame
        """
        # 按指定的指标进行排序
        sorted_df = metrics_df.sort_values(by=metric, ascending=ascending)
        
        # 计算需要选择的行数
        top_n = int(len(sorted_df) * (percent / 100))
        
        # 选取前百分之几的周期组合
        selected_cycles = sorted_df.head(top_n)
        
        # 返回选定的周期组合及其得分
        return selected_cycles

    def _save_decomposition_result(self, scenario, decomposition_results):
        """
        保存时间序列分解结果到指定路径。

        :param scenario: 当前场景名称
        :param decomposition_results: 分解后的结果字典，键为周期组合，值为DataFrame
        """
        output_dir = os.path.join(self.output_dir_base, scenario, 'decomposition')
        os.makedirs(output_dir, exist_ok=True)

        for period_combination, df in decomposition_results.items():
            period_str = '_'.join(map(str, period_combination))
            output_path = os.path.join(output_dir, f'cycle_{period_str}.csv')
            df.to_csv(output_path, index=False)
            print(f"Decomposition result saved to {output_path}")

    def _save_projection_result(self, scenario, projection_results):
        """
        保存时间序列投影结果到指定路径。

        :param scenario: 当前场景名称
        :param projection_results: 带有投影数据的结果字典，键为周期组合，值为带有投影数据的 DataFrame
        """
        output_dir = os.path.join(self.output_dir_base, scenario, 'projection')
        os.makedirs(output_dir, exist_ok=True)

        for period_combination, df in projection_results.items():
            period_str = '_'.join(map(str, period_combination))
            output_path = os.path.join(output_dir, f'cycle_{period_str}.csv')
            df.to_csv(output_path, index=False)
            print(f"Projection result saved to {output_path}")

    def _save_metrics(self, scenario, final_df):
        """
        保存合并后的基础指标和衍生指标到指定路径。

        :param scenario: 当前场景名称
        :param final_df: 合并后的指标 DataFrame
        """
        output_dir = os.path.join(self.output_dir_base, scenario, 'metrics')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics.csv')
        final_df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")

    def _generate_visualization(self, scenario, decomposition_results, final_metrics_df, projection_results, selected_cycles_df):
        """
        生成并保存结果的可视化图表，包括分解结果、指标热图、以及投影图。
        
        :param scenario: 当前场景名称
        :param decomposition_results: 分解后的结果字典，键为周期组合，值为DataFrame
        :param final_metrics_df: 合并后的指标 DataFrame
        :param projection_results: 带有投影数据的结果字典，键为周期组合，值为带有投影数据的 DataFrame
        :param selected_cycles_df: 包含选择的周期组合及其指标的 DataFrame
        """
        output_dir = os.path.join(self.output_dir_base, scenario, 'figures')
        visualizer = DecompositionVisualizer(output_dir=output_dir)
        
        # 可视化分解结果
        visualizer.plot_decomposition_results(decomposition_results)
        
        # 可视化基础指标的热图
        # visualizer.plot_heatmap(final_metrics_df, self.frequency)
        
        # 可视化投影结果
        # visualizer.plot_multiple_projections(selected_cycles_df, projection_results, title="Projection with Selected Cycles")
        
        # 可视化衍生指标的三维mesh图
        # visualizer.plot_mesh(final_metrics_df)

    def run(self):
        """
        执行实验，根据场景和配置文件中的设定。
        """
        model = DecompositionModel()
        projection_model = ProjectionModel(end_date='2035-01-01', frequency=self.frequency)

        for scenario in self.scenarios:
            print(f"Running scenario {scenario}")
            
            periods = self.config['scenarios'][scenario]['periods'][self.frequency]

            # 执行分解并返回所有结果
            decomposition_results = self._execute_combinations(periods, model, scenario_results=[])

            # 计算并合并所有基础指标和衍生指标
            final_metrics_df = self._calculate_metrics(decomposition_results, BasicMetricsCalculator(), DerivedMetricsCalculator())

            # 选择顶级周期组合
            selected_cycles_df = self.select_top_cycles(final_metrics_df, metric='acf', percent=10, ascending=False)

            # 处理分解结果，生成带有投影数据的结果
            projection_results = self._process_projections(decomposition_results, projection_model)

            # 存储结果
            self._save_metrics(scenario, final_metrics_df)
            self._save_decomposition_result(scenario, decomposition_results)
            self._save_projection_result(scenario, projection_results)

            # 如果需要可视化
            if self.visualize and scenario=='cross_year':
                self._generate_visualization(scenario, decomposition_results, final_metrics_df, projection_results, selected_cycles_df)

if __name__ == "__main__":
    import os

    # 设置工作目录为项目的根目录
    os.chdir('E:/MultiPathogen')
    print(f"Current working directory: {os.getcwd()}")

    # runner = ExperimentRunner(
    #     pathogen="flu",
    #     city="Suzhou",
    #     # scenarios=["baseline", "cross_year", "complex"],
    #     scenarios=["baseline", "cross_year"],
    #     # scenarios=["baseline"],
    #     # scenarios=["cross_year"],
    #     # scenarios=["complex"],
    #     parallel=True,
    #     visualize=True
    # )
    # runner.run()

    # for city in ['Beijing', 'Guangzhou', 'Wuhan', 'Xian', 'Lanzhou', 'Suzhou', 'Wenzhou', 'Yunfu']:
    # for city in ['Beijing', 'Guangzhou', 'Wuhan', 'Lanzhou', 'Suzhou', 'Wenzhou', 'Yunfu']:
    for city in ['Beijing']:
        for pathogen in ['flu', 'rsv']:
            print (city, pathogen)
            print ('----------------------------')

            runner = ExperimentRunner(
                pathogen=pathogen,
                city=city,
                scenarios=["cross_year"],
                parallel=True,
                visualize=True
            )
            runner.run()

            # try:
            #     runner = ExperimentRunner(
            #         pathogen=pathogen,
            #         city=city,
            #         scenarios=["cross_year"],
            #         parallel=True,
            #         visualize=True
            #     )
            #     runner.run()
            # except:
            #     pass


