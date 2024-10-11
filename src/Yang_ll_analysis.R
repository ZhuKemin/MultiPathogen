# analysis.R

library(tidyverse)
library(forecast)
library(data.table)

run_analysis <- function(input_csv, output_dir="output") {
  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  
  # Load data
  data <- read_csv(input_csv, col_names = FALSE)
  
  # Define cycles
  cycle1 <- seq(10 * 30, 14 * 30, 30)
  cycle2 <- seq(18 * 30, 36 * 30, 30)
  test_grid_mstl <- CJ(cycle1 = cycle1, cycle2 = cycle2) %>% mutate(ll = as.numeric(NA))

  # MSTL analysis loop
  for (i in 1:nrow(test_grid_mstl)) {
    ts_mstl <- msts(data$X2, seasonal.periods = unlist(test_grid_mstl[i, 1:2]), start = 2010)
    model_mstl <- mstl(ts_mstl)
    model_mstl %>% 
      data.table %>% 
      mutate(predicted = Trend + get(paste0("Seasonal", test_grid_mstl$cycle1[i])) + 
               get(paste0("Seasonal", test_grid_mstl$cycle2[i]))) %>%
      rename(observed = Data) %>% 
      mutate(predicted = if_else(predicted < 0, 0.01, predicted),
             ll = dpois(round(observed), predicted, log = TRUE)) %>%
      pull(ll) %>% 
      sum -> test_grid_mstl[i, "ll"]
  }

  # Save results
  output_path <- file.path(output_dir, paste0(basename(input_csv), "_mstl_results.csv"))
  write.csv(test_grid_mstl, output_path, row.names = FALSE)
  
  return(output_path)
}

# Example call (for testing in R)
# run_analysis("data/raw/flu/some_flu_data.csv")
