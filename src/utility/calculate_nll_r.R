# src/utility/calculate_nll_r.R

# 加载必要的包
library(fitdistrplus)

calculate_gamma_nll <- function(df_path) {
  # 读取数据
  df <- read.csv(df_path)

  # 提取观测值和预测值
  observed <- df$observation
  predicted <- df$observation - df$remainder

  # 修正为非负数，防止log(0) 或负数问题
  observed[observed <= 0] <- 1e-10
  predicted[predicted <= 0] <- 1e-10

  # 使用观测值拟合 Gamma 分布
  fit <- fitdist(observed, "gamma")

  # 提取拟合的形状参数和尺度参数
  shape <- fit$estimate["shape"]
  scale <- fit$estimate["rate"]  # R 中的 Gamma 分布 rate 是 1/scale

  # 使用拟合的参数计算预测值的对数似然
  log_likelihoods <- dgamma(predicted, shape=shape, scale=1/scale, log=TRUE)

  # 检查 log_likelihoods 是否有 NaN
  if (any(is.na(log_likelihoods))) {
    print("Log-likelihoods contain NaN values.")
  }

  # 计算 NLL
  nll <- -sum(log_likelihoods, na.rm=TRUE)

  # 输出最终的 NLL
  cat("Final NLL:", nll, "\n")
}

# 调用函数
args <- commandArgs(trailingOnly = TRUE)
df_path <- args[1]
calculate_gamma_nll(df_path)
