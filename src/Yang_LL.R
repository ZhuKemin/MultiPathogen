library(tidyverse)
library(forecast)
library(data.table)

dir_flu <- "data/raw/flu/"

fn_flu <- paste0(dir_flu, list.files(dir_flu)) %>% 
  map(list.files, pattern = ".csv")

paste0(paste0(dir_flu, list.files(dir_flu), "/"), fn_flu) %>% 
  map(read_csv, col_names = F) %>% 
  setNames(fn_flu) -> data_flu

#### CHEN, Xi'an, Flu ####
cycle1 <- seq(10*30, 14*30, 30)
cycle2 <- seq(18*30,36*30, 30)
test_grid_mstl <- CJ(cycle1 = cycle1, 
                             cycle2 = cycle2) %>% 
  mutate(ll = as.numeric(NA))

for(i in 1:nrow(test_grid_mstl)){
  ts_mstl <- msts(data_flu[[1]]$X2, seasonal.periods = unlist(test_grid_mstl[i,1:2]), start = 2010)
  model_mstl <- mstl(ts_mstl)
  model_mstl %>% 
    data.table %>% 
    mutate(predicted = Trend + get((paste0("Seasonal", test_grid_mstl$cycle1[i]))) + get((paste0("Seasonal",test_grid_mstl$cycle2[i])))) %>% 
    rename(observed = Data) %>% 
    mutate(predicted = if_else(predicted < 0, 0.01, predicted)) %>% 
    mutate(ll = dpois(round(observed),  predicted, log = T)) %>% 
    pull(ll) %>% 
    sum -> test_grid_mstl[i,"ll"]
}

cycle1 <- seq(10*30, 14*30, 30)
test_grid_stl <- CJ(cycle1 = cycle1) %>% 
  mutate(cycle2 = 0,
         ll = as.numeric(NA))

for(i in 1:nrow(test_grid_stl)){
  ts_stl <- ts(data_flu[[1]]$X2, frequency = test_grid_stl$cycle1[i], start = 2010)
  model_stl <- stl(ts_stl, s.window = "periodic")
  model_stl$time.series %>% 
    data.table %>% 
    mutate(predicted = seasonal + trend,
           observed = seasonal + trend + remainder) %>% 
    mutate(ll = dpois(round(observed),  predicted, log = T)) %>% 
    pull(ll) %>% 
    sum -> test_grid_stl[i,"ll"]
}

test_grid_mstl %>% 
  mutate(model = "mstl") %>% 
  bind_rows(test_grid_stl %>% 
              mutate(model = "stl")) -> res

test_grid_mstl %>% 
  ggplot(., aes(x = ll)) +
  geom_density() +
  facet_wrap(~cycle1) +
  geom_vline(data = test_grid_stl,
             aes(xintercept = ll),
             color = "red")

test_grid_mstl %>% 
  mutate(rank = rank(-ll),
         flag = rank <= 10) %>% 
  ggplot(., aes(x = cycle1, y = cycle2, fill = ll, color = flag)) +
  geom_tile(size = 0.5) +
  scale_color_manual(values = c("black", "red"))

#### LIU, Lanzhou, Flu ####
cycle1 <- seq(10*4, 14*4, 4)
cycle2 <- seq(18*4, 32*4, 4)
test_grid_mstl <- CJ(cycle1 = cycle1, 
                     cycle2 = cycle2) %>% 
  mutate(ll = as.numeric(NA))

for(i in 1:nrow(test_grid_mstl)){
  ts_mstl <- msts(data_flu[[2]]$X2, seasonal.periods = unlist(test_grid_mstl[i,1:2]), start = 2016)
  model_mstl <- mstl(ts_mstl)
  model_mstl %>% 
    data.table %>% 
    mutate(predicted = Trend + get((paste0("Seasonal", test_grid_mstl$cycle1[i]))) + get((paste0("Seasonal",test_grid_mstl$cycle2[i])))) %>% 
    rename(observed = Data) %>% 
    mutate(predicted = if_else(predicted < 0, 0.01, predicted)) %>% 
    mutate(ll = dpois(round(observed),  predicted, log = T)) %>% 
    pull(ll) %>% 
    sum -> test_grid_mstl[i,"ll"]
}

cycle1 <- seq(10*4, 14*4, 4)
test_grid_stl <- CJ(cycle1 = cycle1) %>% 
  mutate(cycle2 = 0,
         ll = as.numeric(NA))

for(i in 1:nrow(test_grid_stl)){
  ts_stl <- ts(data_flu[[2]]$X2, frequency = test_grid_stl$cycle1[i], start = 2016)
  model_stl <- stl(ts_stl, s.window = "periodic")
  model_stl$time.series %>% 
    data.table %>% 
    mutate(predicted = seasonal + trend,
           observed = seasonal + trend + remainder,
           predicted = if_else(predicted < 0, 0.01, predicted)) %>% 
    mutate(ll = dpois(round(observed),  predicted, log = T)) %>% 
    pull(ll) %>% 
    sum -> test_grid_stl[i,"ll"]
}

test_grid_mstl %>% 
  mutate(model = "mstl") %>% 
  bind_rows(test_grid_stl %>% 
              mutate(model = "stl")) -> res

test_grid_mstl %>% 
  ggplot(., aes(x = ll)) +
  geom_density() +
  facet_wrap(~cycle1) +
  geom_vline(data = test_grid_stl,
             aes(xintercept = ll),
             color = "red")

res %>% 
  dplyr::filter(cycle1 == 52) %>% 
  mutate()

test_grid_mstl %>% 
  mutate(rank = rank(-ll),
         flag = rank <= 10) %>% 
  ggplot(., aes(x = cycle1, y = cycle2, fill = ll, color = flag)) +
  geom_tile(size = 0.5) +
  scale_color_manual(values = c("black", "red"))
