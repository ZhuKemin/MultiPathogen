suppressMessages(suppressWarnings({
  library(forecast)
}))

# 从标准输入中读取数据
ts_data <- read.csv(file("stdin"), header = TRUE)

# 获取命令行参数，所有的周期长度
periods <- as.numeric(commandArgs(trailingOnly = TRUE))

# 将数据转换为时间序列对象
msts_data <- msts(
  ts_data$value,
  seasonal.periods = periods,
  start = as.Date(ts_data$date[1])
)

# 执行MSTL分解
mstl_result <- mstl(msts_data, s.window = 'periodic', iterate = 50, t.degree = 0)

# 提取分解后的分量
trend <- mstl_result[, "Trend"]
seasonal_columns <- grep("^Seasonal", colnames(mstl_result), value = TRUE)
seasonal <- mstl_result[, seasonal_columns, drop = FALSE]
remainder <- mstl_result[, "Remainder"]

# 组合结果
result <- data.frame(
  date = ts_data$date, 
  observation = ts_data$value,
  trend = trend,
  remainder = remainder
)

# 将所有的季节性分量合并到结果中
result <- cbind(result, seasonal)

# 按顺序排列列：日期，观测值，趋势，周期，残差
ordered_columns <- c("date", "observation", "trend", seasonal_columns, "remainder")
result <- result[, ordered_columns]

# 输出结果到标准输出
write.csv(result, row.names = FALSE)
