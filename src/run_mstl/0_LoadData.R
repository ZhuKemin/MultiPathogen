library(tidyverse)
library(forecast)
library(data.table)
library(yaml)

dir_flu <- "data/raw/flu/"
dir_rsv <- "data/raw/rsv/"

fn_flu <- paste0(dir_flu, list.files(dir_flu)) %>% 
  map(list.files, pattern = ".csv")
fn_rsv <- paste0(dir_rsv, list.files(dir_rsv)) %>% 
  map(list.files, pattern = ".csv")

paste0(paste0(dir_flu, list.files(dir_flu), "/"), fn_flu) %>% 
  map(read_csv, col_names = F) %>% 
  setNames(gsub(".csv", "",fn_flu)) -> data_flu

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
  setNames(dir_rsv_file) -> data_rsv

yaml_data <- yaml.load_file("data/metadata.yaml")