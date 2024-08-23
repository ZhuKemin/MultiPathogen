import pandas as pd
from datetime import datetime, timedelta
import os

class WeeklyToMonthlyAggregator:
    def __init__(self, file_name, start_date):
        self.file_name = file_name
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
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
            # 计算每周的起始日期
            week_num = int(self.data.iloc[i, 0])  # 转换为标准整数
            week_start_date = current_date + timedelta(weeks=week_num - 1)
            weekly_data.append([week_start_date, self.data.iloc[i, 1]])
        
        weekly_df = pd.DataFrame(weekly_data, columns=["Start_Date", "Value"])
        weekly_df["Month"] = weekly_df["Start_Date"].dt.to_period("M")
        
        monthly_aggregated = weekly_df.groupby("Month").sum()["Value"].reset_index()
        monthly_aggregated["Year-Month"] = monthly_aggregated["Month"].dt.strftime("%Y-%m")
        monthly_aggregated["Positive Cases"] = monthly_aggregated["Value"].apply(lambda x: max(round(x), 0))
        
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
    # file_name = "Liu-2022-Lanzhou"
    # start_date = "2016-01-01"  # 假设数据从2016年的第一周开始
    
    file_name = "Yan-2024-Guangzhou"
    start_date = "2010-10-04"  # 假设数据从2016年的第一周开始

    aggregator = WeeklyToMonthlyAggregator(file_name, start_date)
    result = aggregator.process()
    if result is not None:
        print(f"Processed {file_name}, results saved.")
    else:
        print(f"Failed to process {file_name}.")
