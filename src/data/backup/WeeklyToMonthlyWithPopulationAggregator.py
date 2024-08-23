import pandas as pd
from datetime import datetime, timedelta
import os

class WeeklyToMonthlyWithPopulationAggregator:
    def __init__(self, file_name, start_date, population):
        self.file_name = file_name
        self.start_date = datetime.strptime(start_date, "%Y-%W-%w")
        self.population = population
        self.data_file = os.path.join(file_name, f"{file_name}.csv")
        self.data = None
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_file, header=None, sep=None, engine='python')
            if self.data.shape[1] == 2:  # 确保数据有两列
                self.data[0] = self.data[0].astype(int)
                self.data[1] = self.data[1].astype(float)
            else:
                print("Error: The data does not have exactly two columns.")
                return False
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        except ValueError as e:
            print(f"Error: {e}")
            return False
        return True
    
    def aggregate_to_monthly(self):
        weekly_data = []
        current_date = self.start_date
        
        for i in range(len(self.data)):
            week_num = int(self.data.iloc[i, 0])
            week_start_date = current_date + timedelta(weeks=week_num - 1)
            # 计算每周的阳性病例数（每十万人）* 总人口（单位：百万）
            positive_cases = self.data.iloc[i, 1] * (self.population / 100000)
            weekly_data.append([week_start_date, positive_cases])
        
        weekly_df = pd.DataFrame(weekly_data, columns=["Start_Date", "Positive Cases"])
        weekly_df["Month"] = weekly_df["Start_Date"].dt.to_period("M")
        
        monthly_aggregated = weekly_df.groupby("Month").sum()["Positive Cases"].reset_index()
        monthly_aggregated["Year-Month"] = monthly_aggregated["Month"].dt.strftime("%Y-%m")
        monthly_aggregated["Positive Cases"] = monthly_aggregated["Positive Cases"].apply(lambda x: max(round(x), 0))
        
        return monthly_aggregated[["Year-Month", "Positive Cases"]]
    
    def save_results(self, result):
        output_file = os.path.join("..", f"{self.file_name}-Cases.csv")
        result.to_csv(output_file, index=False)
    
    def process(self):
        if not self.load_data():
            return None
        result = self.aggregate_to_monthly()
        if result is not None:
            self.save_results(result)
        return result

if __name__ == "__main__":
    file_name = "Wu-2018-Beijing"
    start_date = "2007-27-1"  # 假设数据从2007年的第27周开始
    population = 20000000  # 总人口数
    population = 2000000
    
    aggregator = WeeklyToMonthlyWithPopulationAggregator(file_name, start_date, population)
    result = aggregator.process()
    if result is not None:
        print(f"Processed {file_name}, results saved.")
    else:
        print(f"Failed to process {file_name}.")
