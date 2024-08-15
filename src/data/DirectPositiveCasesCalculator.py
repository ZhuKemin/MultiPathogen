import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

class DirectPositiveCasesCalculator:
    def __init__(self, folder_name, start_date):
        self.folder_name = folder_name
        self.start_date = datetime.strptime(start_date, "%Y-%m")
        self.data_file = os.path.join(folder_name, f"{folder_name}.csv")
        self.data = None
    
    def load_data(self):
        try:
            # 尝试使用多种分隔符读取数据
            for sep in [',', '\t', ';', ' ']:
                try:
                    self.data = pd.read_csv(self.data_file, header=None, sep=sep)
                    if self.data.shape[1] == 2:  # 确保数据有两列
                        self.data[0] = self.data[0].astype(int)
                        self.data[1] = self.data[1].astype(float)
                        break  # 成功读取并转换数据后退出循环
                except ValueError:
                    continue
            if self.data is None or self.data.shape[1] != 2:
                raise ValueError("Unable to parse the data file with known delimiters or incorrect number of columns.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        except ValueError as e:
            print(f"Error: {e}")
            return False
        return True
    
    def calculate_positive_cases(self):
        positive_cases = []
        current_date = self.start_date
        
        for i in range(len(self.data)):
            positive_count = round(self.data.iloc[i, 1])
            positive_count = max(positive_count, 0)  # 将负数转换为零
            
            positive_cases.append([current_date.strftime("%Y-%m"), positive_count])
            current_date += relativedelta(months=1)
        
        return pd.DataFrame(positive_cases, columns=["Year-Month", "Positive Cases"])
    
    def save_results(self, result):
        output_file = os.path.join("..", f"{self.folder_name}-Cases.csv")
        result.to_csv(output_file, index=False)
    
    def process(self):
        if not self.load_data():
            return None
        result = self.calculate_positive_cases()
        if result is not None:
            self.save_results(result)
        return result

if __name__ == "__main__":
    folders_and_dates = [
        # ("Yu-2019-Suzhou"    , "2011-10"),
        ("Liu-2023-Wuhan"    , "2018-05"),
        ("Peng-2016-Yunfu"   , "2010-01"),
        ("Zhong-2016-Wenzhou", "2010-01"),
    ]
    
    for folder, date in folders_and_dates:
        calculator = DirectPositiveCasesCalculator(folder, date)
        result = calculator.process()
        if result is not None:
            print(f"Processed {folder}, results saved.")
        else:
            print(f"Failed to process {folder}.")
