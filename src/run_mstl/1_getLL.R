# @study name: 
# (1) disease type
# (2) data type
# (3) cycle 1 and cycle 2 assumption will included in this
library(pacman)
p_load(magrittr)

source("src/run_mstl/0_LoadData.R")

get_LL <- function(disease = NULL,
                   city = NULL){
  
  study_name = yaml_data[[city]][[disease]]$name
  data_type = yaml_data[[city]][[disease]]$data_type
  cat("Study currently in use is:", study_name)
  cat("Data type currently in use is:", data_type)
  
  if(disease == "flu") {tmp <- data_flu[[study_name]]}
  if(disease == "rsv") {tmp <- data_rsv[[study_name]]}
  
  n_steps <- nrow(tmp)
  step_size_char <- yaml_data[[city]][[disease]]$raw_data_freq
  step_size_num <- case_when(step_size_char == "daily" ~ 30,
                             step_size_char == "weekly" ~ 4,
                             step_size_char == "monthly"~ 1)
  cycle_max <- round(n_steps/2)
  
  cycle1 <- seq(10*step_size_num, 14*step_size_num, step_size_num)
  
  if(cycle_max > 36*step_size_num){
    cycle2 <- seq(18*step_size_num, 
                  36*step_size_num, 
                  step_size_num)
  } else{
    cycle2 <- seq(18*step_size_num, 
                  floor(cycle_max/step_size_num)*step_size_num - 1, 
                  step_size_num)  
  }
  
  if(data_type == "cases" & disease == "flu"){
    cycle1 %>% 
      map(~ts(log(tmp$X2), frequency = .)) %>% 
      map(stl, s.window = "periodic") %>% 
      map(~.$time.series) %>% 
      map(data.table) %>% 
      map(mutate,
          predicted = exp(seasonal + trend),
          observed = exp(seasonal + trend + remainder)) %>% 
      setNames(cycle1) %>% 
      bind_rows(.id = "length_cycle1") %>% 
      mutate(model = "stl",
             ll = dpois(round(observed, 0),
                        predicted,
                        log = T)) %>% 
      group_by(length_cycle1) %>% 
      mutate(ll_sum = sum(ll)) -> output_stl
    
    map2(CJ(cycle1, cycle2) %>% pull(cycle1), 
         CJ(cycle1, cycle2) %>% pull(cycle2),
         ~msts(data = log(tmp$X2), 
               seasonal.periods = c(.x, .y))) %>% 
      map(mstl, iterate = 3) %>% 
      map(data.table) %>% 
      map(setNames,
          c("Data", "Trend", "component_cycle1", "component_cycle2", "remainder")) %>% 
      bind_rows(.id = "combo_index") -> output_mstl
    
    CJ(length_cycle1 = cycle1, length_cycle2 = cycle2) %>% 
      rownames_to_column(var = "combo_index") %>% 
      left_join(output_mstl, by = "combo_index") %>% 
      mutate(predicted = exp(Trend + component_cycle1 + component_cycle2),
             observed = exp(Data)) %>% 
      mutate(ll = dpois(round(observed, 0),
                        predicted,
                        log = T)) %>% 
      group_by(combo_index, length_cycle1, length_cycle2) %>% 
      mutate(ll_sum = sum(ll),
             model = "mstl") -> output2_mstl
    
    output_stl[,c("length_cycle1", "predicted", "observed", 
                  "model", "ll", "ll_sum")] %>% 
      mutate(length_cycle2 = as.numeric(NA),
             length_cycle1 = as.numeric(length_cycle1)) %>% 
      ungroup %>% 
      dplyr::select(c("length_cycle1", "length_cycle2","predicted", "observed", 
                      "model", "ll", "ll_sum")) %>%
      bind_rows(output2_mstl %>% ungroup %>% 
                  dplyr::select(c("length_cycle1", "length_cycle2","predicted", "observed", 
                                  "model", "ll", "ll_sum"))) %>% 
      dplyr::select(length_cycle1,
                    length_cycle2,
                    model,
                    ll_sum) %>% 
      distinct() -> res
  }
  
  if(data_type == "cases" & disease == "rsv"){
    
    cycle1 %>% 
      map(~ts((tmp$X2), frequency = .)) %>% 
      map(stl, s.window = "periodic") %>% 
      map(~.$time.series) %>% 
      map(data.table) %>% 
      map(mutate,
          predicted = (seasonal + trend),
          observed = (seasonal + trend + remainder),
          predicted = if_else(predicted < 0, 0.01, predicted)) %>% 
      setNames(cycle1) %>% 
      bind_rows(.id = "length_cycle1") %>% 
      mutate(model = "stl",
             ll = dpois(round(observed, 0),
                        predicted, 
                        log = T)) %>% 
      group_by(length_cycle1) %>% 
      summarise(ll_sum = sum(ll)) %>% 
      mutate(model = "stl",
             length_cycle2 = as.numeric(NA),
             length_cycle1 = as.numeric(length_cycle1)) -> output_stl
    
    map2(CJ(cycle1, cycle2) %>% pull(cycle1), 
         CJ(cycle1, cycle2) %>% pull(cycle2),
         ~msts(data = (tmp$X2), 
               seasonal.periods = c(.x, .y))) %>% 
      map(mstl, iterate = 3) %>% 
      map(data.table) %>% 
      map(setNames,
          c("Data", "Trend", "component_cycle1", "component_cycle2", "remainder")) %>% 
      bind_rows(.id = "combo_index") -> output_mstl
    
    CJ(length_cycle1 = cycle1, length_cycle2 = cycle2) %>% 
      rownames_to_column(var = "combo_index") %>% 
      left_join(output_mstl, by = "combo_index") %>% 
      mutate(predicted = (Trend + component_cycle1 + component_cycle2),
             observed = (Data),
             predicted = if_else(predicted < 0, 0.01, predicted)) %>% 
      mutate(ll = dpois(round(observed, 0),
                        predicted,
                        log = T)) %>% 
      group_by(combo_index, length_cycle1, length_cycle2) %>% 
      summarise(ll_sum = sum(ll)) %>% 
      mutate(model = "mstl") -> output2_mstl
    
    output2_mstl %>% 
      ungroup %>% 
      dplyr::select(length_cycle1,
                    length_cycle2,
                    model,
                    ll_sum) %>% 
      bind_rows(output_stl %>%
                  dplyr::select(length_cycle1,
                                length_cycle2,
                                model,
                                ll_sum))-> res
  }
  
  if(data_type == "rates"){
    cycle1 %>% 
      map(~ts(log(tmp$X2/100), frequency = .)) %>% 
      map(stl, s.window = "periodic") %>% 
      map(~.$time.series) %>% 
      map(data.table) %>% 
      map(mutate,
          predicted = exp(seasonal + trend),
          predicted_logged = seasonal + trend,
          observed = exp(seasonal + trend + remainder),
          observed_logged = (seasonal + trend + remainder),
          ) %>% 
      setNames(cycle1) -> output_stl
  
    map(.x = output_stl,
         .f = function(.x) {
           .x %>% 
             mutate(ll = dnorm(predicted_logged, 
                               mean = observed_logged, 
                               sd   = sd(log(tmp$X2/100)),
                               log  = T))
         } 
         ) %>% 
      map(rownames_to_column,
          var = "t") %>% 
      map(mutate, t = as.numeric(t)) %>% 
      bind_rows(.id = "length_cycle1") %>% 
      group_by(length_cycle1) %>% 
      summarise(ll_sum = sum(ll)) %>% 
      mutate(length_cycle2 = as.numeric(NA),
             model = "stl") -> res_stl
    
    map2(CJ(cycle1, cycle2) %>% pull(cycle1), 
         CJ(cycle1, cycle2) %>% pull(cycle2),
         ~msts(data = log(tmp$X2/100), 
               seasonal.periods = c(.x, .y))) %>% 
      map(mstl, iterate = 3) %>% 
      map(data.table) %>% 
      map(setNames,
          c("Data", 
            "Trend", 
            "component_cycle1", 
            "component_cycle2", 
            "remainder")) %>% 
      map(mutate,
          observed = exp(Data),
          predicted = exp(Trend + component_cycle1 + component_cycle2),
          predicted_logged = Trend + component_cycle1 + component_cycle2,
          observed_logged = Data) -> output_mstl
  
    map(.x = output_mstl,
         .f = function(.x) {
           .x %>% 
             mutate(ll = dnorm(predicted_logged, 
                               mean = observed_logged, 
                               sd = sd(log(tmp$X2/100)),
                               log = T))
         } 
    ) %>% 
      bind_rows(.id = "combo_index") %>% 
      left_join(CJ(length_cycle1 = cycle1,
                   length_cycle2 = cycle2) %>% 
                  rownames_to_column(var = "combo_index"),
                by = "combo_index") %>% 
      mutate(model = "mstl") %>% 
      group_by(length_cycle1, length_cycle2, model) %>% 
      summarise(ll_sum = sum(ll)) %>% 
      dplyr::select(colnames(res_stl)) -> res_mstl
    
    res <- bind_rows(res_stl %>% mutate(length_cycle1 = as.numeric(length_cycle1)), 
                     res_mstl)
  
  }
  
  return(res)
}


draw_LL_comparison <- function(city = NULL,
                               disease = NULL){
  
  LL_table <-  get_LL(disease = disease, city = city) 
  
  LL_table %>% 
    dplyr::filter(model == "mstl") %>% 
    ggplot(., aes(x = ll_sum)) + 
    geom_density() +
    facet_wrap(~length_cycle1) +
    geom_vline(data = LL_table %>% 
                 dplyr::filter(model == "stl"),
               aes(xintercept = ll_sum),
               color = "red") +
    labs(x = "LogLik", y = "density",
         title = paste0("city = ", city, "; disease = ", disease)) -> p
  
  return(p)

  }

get_LL(disease = "flu", city = "Xian") # works
get_LL(disease = "flu", city = "Beijing") # works
get_LL(disease = "flu", city = "Guangzhou") # works
get_LL(disease = "flu", city = "Lanzhou") # works
get_LL(disease = "flu", city = "Wenzhou") # works
get_LL(disease = "flu", city = "Wuhan") # works
get_LL(disease = "flu", city = "Yunfu") # works
# get_LL(disease = "flu", city = "Suzhou")

get_LL(disease = "rsv", city = "Xian") # works
get_LL(disease = "rsv", city = "Beijing") # works
get_LL(disease = "rsv", city = "Guangzhou") # works
# get_LL(disease = "rsv", city = "Lanzhou")
get_LL(disease = "rsv", city = "Wenzhou") # works
get_LL(disease = "rsv", city = "Wuhan") # works
get_LL(disease = "rsv", city = "Yunfu") # works
get_LL(disease = "rsv", city = "Suzhou") # works

for(i in 1:7){
  
  city <- c("Xian", "Beijing", "Guangzhou", "Wenzhou",
            "Wuhan", "Yunfu", "Suzhou")
  
  draw_LL_comparison(city = city[i],
                     disease = "rsv") -> p_save
  
  ggsave(paste0(paste0("figs/diagnostics/","rsv","-",city[i],".png")),
         p_save)
  
}


for(i in 1:7){
  
  city <- c("Xian", "Beijing", "Guangzhou", "Lanzhou", "Wenzhou",
            "Wuhan", "Yunfu")
  
  draw_LL_comparison(city = city[i],
                          disease = "flu") -> p_save
  
  ggsave(paste0(paste0("figs/diagnostics/","flu","-",city[i],".png")),
         p_save)

  }
