library(tidyverse)
library(forecast)
library(data.table)
library(yaml)

yaml_data <- yaml.load_file("data/metadata.yaml")

dir_flu <- "data/raw/flu/"
dir_rsv <- "data/raw/rsv/"

fn_flu <- paste0(dir_flu, list.files(dir_flu)) %>% 
  map(list.files, pattern = ".csv")
fn_rsv <- paste0(dir_rsv, list.files(dir_rsv)) %>% 
  map(list.files, pattern = ".csv")

paste0(paste0(dir_flu, list.files(dir_flu), "/"), fn_flu) %>% 
  map(read_csv, col_names = F) %>% 
  setNames(gsub(".csv", "",fn_flu)) -> data_flu

yaml_data$Guangzhou$flu$raw_data_freq <- "weekly"
data_flu$`Yan-2024-Guangzhou` %<>% 
  mutate(X2 = if_else(X2 < 0, 0.01, X2))

fn_rsv %>% 
  setNames(list.files(dir_rsv)) %>% 
  map(data.frame) %>% 
  bind_rows(., .id = "folder_name") %>% 
  mutate(root = dir_rsv) %>% 
  rename(fn = `.x..i..`) %>% 
  mutate(dir_all = paste0(root, 
                          "/",
                          folder_name,
                          "/",
                          fn)) %>% 
  pull(dir_all) -> dir_rsv_file

dir_rsv_file %>% 
  map(read_csv, col_names = F) %>% 
  setNames(dir_rsv_file) -> data_rsv_raw

data_rsv <- list()
data_rsv[[yaml_data$Beijing$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Cui-2013-Beijing/Cui-2013-Beijing-Curve.csv`
yaml_data$Beijing$rsv$data_type <- "rates"

data_rsv[[yaml_data$Wuhan$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Hu-2023-Wuhan/Hu-2023-Wuhan-Curve.csv`
yaml_data$Wuhan$rsv$data_type <- "rates"

data_rsv[[yaml_data$Wenzhou$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Jie-2011-Wenzhou/Jie-2011-Wenzhou.csv`
yaml_data$Wenzhou$rsv$data_type 

data_rsv[[yaml_data$Xian$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Lei-2016-Xian/Lei-2016-Xian.csv`
yaml_data$Xian$rsv$data_type 
data_rsv$`Lei-2016-Xian` %<>% 
  mutate(X2 = if_else(X2 == 0, 0.01, X2))

data_rsv[[yaml_data$Suzhou$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Lu-2015-Suzhou/Lu-2015-Suzhou.csv`
yaml_data$Suzhou$rsv$data_type 

data_rsv[[yaml_data$Yunfu$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Qin-2022-Yunfu/Qin-2022-Yunfu.csv`
yaml_data$Yunfu$rsv$data_type 

data_rsv[[yaml_data$Guangzhou$rsv$name]] <- data_rsv_raw$`data/raw/rsv//Zou-2016-Guangzhou/Zou-2016-Guangzhou-Curve.csv`
yaml_data$Guangzhou$rsv$data_type <- "rates"

# Lanzhou
data_rsv[[yaml_data$Lanzhou$rsv$name]] <- read_csv("data/processed/rsv/Liang-2015-Lanzhou-Cases.csv") %>% 
  rename(X2 = value)

# Guangzhou
data_rsv_raw$guangzhou