

#### CHEN, Xi'an, Flu ####
cycle1 <- seq(10*30, 14*30, 30)
cycle2 <- seq(18*30, 36*30, 30)

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

#### CUI, Beijing, RSV ####
# the temporal unit is month
data_rsv$`data/raw/rsv//Cui-2013-Beijing/Cui-2013-Beijing-Curve.csv` %>% 
  rename(curve = X2, 
         t1 = X1) %>% 
  bind_cols(data_rsv$`data/raw/rsv//Cui-2013-Beijing/Cui-2013-Beijing-Bar.csv`) %>% 
  rename(denominator = X2,
         t2 = X1) %>% 
  mutate(t1 = round(t1,0), t2 = round(t2,0),
         numerator = denominator*curve/100,
         denominator = round(denominator),
         numerator = round(numerator),
         r = numerator/ denominator,
         r = if_else(r == 0, 0.01, r)) -> Cui_beijing_RSV

library(MASS)

fitdistr(Cui_beijing_RSV$r, 
         dbeta,
         start = list(shape1 = 1,
                      shape2 = 1)) -> params_fitted

params_fitted$estimate[1]/(params_fitted$estimate[1] + params_fitted$estimate[2])

cycle1 <- seq(10, 14, 1)
cycle2 <- seq(18, 29, 1)
test_grid_mstl <- CJ(cycle1 = cycle1, 
                     cycle2 = cycle2) %>% 
  mutate(ll = as.numeric(NA))

ll_mstl_list <- list()
for(i in 1:nrow(test_grid_mstl)){
  ts_mstl <- msts(Cui_beijing_RSV$r, seasonal.periods = unlist(test_grid_mstl[i,1:2]), start = 2007)
  model_mstl <- mstl(ts_mstl)
  model_mstl %>% 
    data.table %>% 
    mutate(predicted = Trend + get((paste0("Seasonal", test_grid_mstl$cycle1[i]))) + get((paste0("Seasonal",test_grid_mstl$cycle2[i])))) %>% 
    rename(observed = Data) %>% 
    mutate(alpha = params_fitted$estimate[1],
           beta = params_fitted$estimate[2],
           predicted = if_else(predicted < 0, 0.01, predicted),
           fix_alpha = ((1-predicted)*alpha/predicted),
           fix_beta = beta*predicted/(1-predicted)) %>% 
    mutate(ll_fix_alpha = dbeta(observed,  
                                shape1 = alpha,
                                shape2 = fix_alpha,
                                log = T),
           ll_fix_beta = dbeta(observed,  
                               shape1 = fix_beta,
                               shape2 = beta,
                               log = T)) %>% 
    dplyr::select(starts_with("ll")) %>% 
    summarise(ll_fix_alpha = sum(ll_fix_alpha),
              ll_fix_beta = sum(ll_fix_beta)) -> ll_mstl_list[[i]]
}

cycle1 <- seq(10, 14, 1)
test_grid_stl <- CJ(cycle1 = cycle1) %>% 
  mutate(cycle2 = 0,
         ll = as.numeric(NA))

ll_stl_list <- list()
for(i in 1:nrow(test_grid_stl)){
  ts_stl <- ts(Cui_beijing_RSV$r, frequency = test_grid_stl$cycle1[i], start = 2007)
  model_stl <- stl(ts_stl, s.window = "periodic")
  model_stl$time.series %>% 
    data.table %>% 
    mutate(predicted = seasonal + trend,
           observed = seasonal + trend + remainder) %>% 
    mutate(alpha = params_fitted$estimate[1],
           beta = params_fitted$estimate[2],
           predicted = if_else(predicted < 0, 0.01, predicted),
           fix_alpha = ((1-predicted)*alpha/predicted),
           fix_beta = beta*predicted/(1-predicted)) %>% 
    mutate(ll_fix_alpha = dbeta(observed,  
                                shape1 = alpha,
                                shape2 = fix_alpha,
                                log = T),
           ll_fix_beta = dbeta(observed,  
                               shape1 = fix_beta,
                               shape2 = beta,
                               log = T)) %>% 
    dplyr::select(starts_with("ll")) %>% 
    summarise(ll_fix_alpha = sum(ll_fix_alpha),
              ll_fix_beta = sum(ll_fix_beta)) -> ll_stl_list[[i]]
}



ll_mstl_list %>% 
  bind_rows() %>% 
  bind_cols(test_grid_mstl) %>% 
  dplyr::select(-ll) %>% 
  mutate(model = "mstl") %>% 
  bind_rows(ll_stl_list %>% 
              setNames(test_grid_stl$cycle1) %>% 
              bind_rows(.id = "cycle1") %>% 
              mutate(model = "stl",
                     cycle1 = as.numeric(cycle1))) %>% 
  dplyr::filter(cycle1 == 10)



test_grid_mstl %>% 
  mutate(model = "mstl") %>% 
  bind_rows(test_grid_stl %>% 
              mutate(model = "stl")) -> res

test_grid_mstl %>% 
  mutate(model = "mstl") %>% 
  left_join(test_grid_stl %>% 
              mutate(model = "stl") %>%
              rename(stl_ll = ll) %>% 
              dplyr::select(-cycle2),
            by = "cycle1") %>% 
  mutate(LLR = ll/stl_ll) %>% 
  dplyr:;

test_grid_mstl %>% 
  ggplot(., aes(x = ll)) +
  geom_density() +
  facet_wrap(~cycle1) +
  geom_vline(data = test_grid_stl,
             aes(xintercept = ll),
             color = "red")


  