import pandas as pd
from datetime import datetime
import os

# TODO: 根据文件结构修改所有的 raw data 预处理程序

class DailyToMonthlyAggregator:
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
        daily_data = []
        current_date = self.start_date
        
        for i in range(len(self.data)):
            date = current_date + pd.Timedelta(days=self.data.iloc[i, 0] - 1)
            daily_data.append([date, self.data.iloc[i, 1]])
        
        daily_df = pd.DataFrame(daily_data, columns=["Date", "Value"])
        daily_df["Month"] = daily_df["Date"].dt.to_period("M")
        
        monthly_aggregated = daily_df.groupby("Month").sum()["Value"].reset_index()
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
    file_name = "Chen-2023-Xian"
    start_date = "2010-01-01"
    
    aggregator = DailyToMonthlyAggregator(file_name, start_date)
    result = aggregator.process()
    if result is not None:
        print(f"Processed {file_name}, results saved.")
    else:
        print(f"Failed to process {file_name}.")
