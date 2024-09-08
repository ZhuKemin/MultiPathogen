suppressMessages(suppressWarnings({
  library(forecast)
}))

# 从标准输入中读取数据
ts_data <- read.csv(file("stdin"), header = TRUE)

# 获取命令行参数，假设第一个参数是周期长度
period <- as.numeric(commandArgs(trailingOnly = TRUE)[1])

# 将数据转换为时间序列对象
ts <- ts(
  ts_data$value,
  frequency = period,
  start = as.Date(ts_data$date[1])
)

# 执行STL分解
stl_result <- stl(ts, s.window = "periodic")

# 提取分解后的分量
trend <- stl_result$time.series[, "trend"]
seasonal <- stl_result$time.series[, "seasonal"]
remainder <- stl_result$time.series[, "remainder"]

# 组合结果
result <- data.frame(
  date = ts_data$date,
  observation = ts_data$value,
  trend = trend,
  seasonal = seasonal,
  remainder = remainder
)

# 按顺序排列列：日期，观测值，趋势，周期，残差
ordered_columns <- c("date", "observation", "trend", "seasonal", "remainder")
result <- result[, ordered_columns]

# 输出结果到标准输出
write.csv(result, row.names = FALSE)
