# Figure 2: are they chosen? are they doing better? is there a disease 
# combination where more complex models are not necessarily doing better?

# Figure 3: annual and interannual component selection: how long are they? 
# how large are these components?

##### cycle1: selected####
LL_flu %>% 
  bind_rows() %>% 
  dplyr::filter(model == "stl") %>% 
  group_by(city) %>% 
  mutate(ll_sum_max = max(ll_sum),
         ll_relative = if_else(ll_sum < 0, ll_sum_max/ll_sum, ll_sum/ll_sum_max),
         ll_similar = sum(ll_relative >= 0.9) - 1,
         # ll_relative = ll_sum_max/ll_sum,
         step_size_standardised = case_when(step_size_char == "daily" ~ length_cycle1/30,
                                            step_size_char == "weekly" ~ length_cycle1/4,
                                            step_size_char == "monthly" ~ length_cycle1),
         acf1_max = max(acf1),
         acf1_relative = acf1/acf1_max,
         acf1_similar = sum(acf1_relative >= 0.9) - 1,
         rank_ll = rank(desc(ll_sum)),
         rank_mae = rank(mae_mean),
         rank_acf = rank(acf1),
         rank_size = rank(desc(size_cycle1)),
         rank_mean = (rank_acf + rank_ll + rank_size)/3,
         rank_mean_min = min(rank_mean)) %>%
  dplyr::filter(rank_mean == rank_mean_min) %>% 
  mutate(cycle1_selected = case_when(city == "Yunfu" ~ 11,
                                     ll_similar == 4 ~ NA,
                                     TRUE ~ 12),
         city = factor(city, levels = city_ordered)) %>% 
  bind_rows(LL_rsv %>% 
              bind_rows() %>% 
              dplyr::filter(model == "stl") %>% 
              group_by(city) %>% 
              mutate(ll_sum_max = max(ll_sum),
                     ll_relative = if_else(ll_sum < 0, ll_sum_max/ll_sum, ll_sum/ll_sum_max),
                     ll_similar = sum(ll_relative >= 0.9) - 1,
                     # ll_relative = ll_sum_max/ll_sum,
                     step_size_standardised = case_when(step_size_char == "daily" ~ length_cycle1/30,
                                                        step_size_char == "weekly" ~ length_cycle1/4,
                                                        step_size_char == "monthly" ~ length_cycle1),
                     acf1_max = max(acf1),
                     acf1_relative = acf1/acf1_max,
                     acf1_similar = sum(acf1_relative >= 0.9) - 1,
                     rank_ll = rank(desc(ll_sum)),
                     rank_mae = rank(mae_mean),
                     rank_acf = rank(acf1),
                     rank_size = rank(desc(size_cycle1)),
                     rank_mean = (rank_acf + rank_ll + rank_size)/3,
                     rank_mean_min = min(rank_mean)) %>%
              dplyr::filter(rank_mean == rank_mean_min) %>%
              mutate(cycle1_selected = case_when(city == "Yunfu" ~ 10,
                                                 ll_similar == 4 ~ NA,
                                                 TRUE ~ 12),
                     city = factor(city, levels = city_ordered))) %>% 
  mutate(disease = factor(disease,
                          levels = c("flu", "rsv"),
                          labels = c("Influenza",
                                     "RSV")))-> p1_tab

p1_tab %>% 
  ggplot(., aes(x = disease, y = city, fill = cycle1_selected)) +
  geom_tile(color = "black", size = 1) +
  labs(x = "", y = "") +
  scale_fill_viridis_c(limits = c(10, 14), option = "magma") + 
  coord_fixed() -> p1

#### cycle2: selected ####
LL_flu %>% 
  bind_rows() %>% 
  dplyr::filter(model == "mstl") %>% 
  group_by(city) %>% 
  mutate(ll_sum_max = max(ll_sum),
         ll_relative = if_else(ll_sum < 0, ll_sum_max/ll_sum, ll_sum/ll_sum_max),
         ll_similar = sum(ll_relative >= 0.9) - 1,
         # ll_relative = ll_sum_max/ll_sum,
         step_size1_standardised = case_when(step_size_char == "daily" ~ length_cycle1/30,
                                            step_size_char == "weekly" ~ length_cycle1/4,
                                            step_size_char == "monthly" ~ length_cycle1),
         step_size2_standardised = case_when(step_size_char == "daily" ~ length_cycle2/30,
                                             step_size_char == "weekly" ~ length_cycle2/4,
                                             step_size_char == "monthly" ~ length_cycle2),
         acf1_max = max(acf1),
         acf1_relative = acf1/acf1_max,
         acf1_similar = sum(acf1_relative >= 0.9) - 1,
         rank_ll = rank(desc(ll_sum)),
         rank_mae = rank(mae_mean),
         rank_acf = rank(acf1),
         rank_size = rank(desc(size_cycle1)),
         rank_mean = (rank_acf + rank_ll + rank_size)/3,
         rank_mean_min = min(rank_mean)) %>%
  dplyr::filter(rank_mean == rank_mean_min) %>% 
  bind_rows(LL_rsv %>% 
              bind_rows() %>% 
              dplyr::filter(model == "mstl") %>% 
              group_by(city) %>% 
              mutate(ll_sum_max = max(ll_sum),
                     ll_relative = if_else(ll_sum < 0, ll_sum_max/ll_sum, ll_sum/ll_sum_max),
                     ll_similar = sum(ll_relative >= 0.9) - 1,
                     # ll_relative = ll_sum_max/ll_sum,
                     step_size1_standardised = case_when(step_size_char == "daily" ~ length_cycle1/30,
                                                        step_size_char == "weekly" ~ length_cycle1/4,
                                                        step_size_char == "monthly" ~ length_cycle1),
                     step_size2_standardised = case_when(step_size_char == "daily" ~ length_cycle2/30,
                                                         step_size_char == "weekly" ~ length_cycle2/4,
                                                         step_size_char == "monthly" ~ length_cycle2),
                     acf1_max = max(acf1),
                     acf1_relative = acf1/acf1_max,
                     acf1_similar = sum(acf1_relative >= 0.9) - 1,
                     rank_ll = rank(desc(ll_sum)),
                     rank_mae = rank(mae_mean),
                     rank_acf = rank(acf1),
                     rank_size = rank(desc(size_cycle1)),
                     rank_mean = (rank_acf + rank_ll + rank_size)/3,
                     rank_mean_min = min(rank_mean)) %>%
              dplyr::filter(rank_mean == rank_mean_min)) %>%  
  mutate(city = factor(city, levels = city_ordered),
         disease = factor(disease,
                          levels = c("flu", "rsv"),
                          labels = c("Influenza",
                                     "RSV"))) -> p2_tab

p2_tab %>% 
  ggplot(., aes(x = disease, y = city, fill = step_size2_standardised)) +
  geom_tile(color = "black", size = 1) +
  scale_fill_viridis_c(option = "magma", limits = c(18, 34)) +
  labs(x = "", y = "") +
  coord_fixed() -> p2

#### panel3: data range analysed ####
data_flu %>% 
  bind_rows(.id = "study") %>% 
  separate(study, into = c("first_author", "year", "city")) %>% 
  group_by(year, city, first_author) %>% 
  summarise(date_min = min(date),
            date_max = max(date)) %>% 
  mutate(date_min = if_else(first_author == "Yan", lubridate::ymd("2010-10-04"), date_min),
         date_max = if_else(first_author == "Yan", lubridate::ymd("2020-01-05"), date_max),
         duration = date_max - date_min,
         disease = "Influenza") %>% 
  bind_rows(data_rsv %>% 
              bind_rows(.id = "study") %>% 
              separate(study, into = c("first_author", "year", "city")) %>% 
              group_by(year, city, first_author) %>% 
              summarise(date_min = min(date),
                        date_max = max(date)) %>% 
              mutate(date_min = if_else(first_author == "Yan", lubridate::ymd("2010-10-04"), date_min),
                     date_max = if_else(first_author == "Yan", lubridate::ymd("2020-01-05"), date_max),
                     duration = date_max - date_min,
                     disease = "RSV")) %>% 
  mutate(city = factor(city, levels = city_ordered),
         duration = as.numeric(duration)) %>% 
  ggplot(., aes(x = disease, y = city, fill = duration)) +
  geom_tile(color = "black", size = 1) +
  scale_fill_viridis_c(option = "magma") +
  coord_fixed() +
  labs(x = "", y = "") -> p3
  
#### panel4: unit ####
LL_flu %>% 
  bind_rows() %>% 
  dplyr::select(city, step_size_char, data_type) %>% 
  distinct() %>% 
  mutate(disease = "Influenza") %>% 
  bind_rows(LL_rsv %>% 
              bind_rows() %>% 
              dplyr::select(city, step_size_char, data_type) %>% 
              distinct() %>% 
              mutate(disease = "RSV")) %>% 
  mutate(city = factor(city, levels = city_ordered),
         step_size_char = factor(step_size_char,
                                 levels = c("daily", "weekly", "monthly"),
                                 labels = c("Daily", "Weekly", "Monthly"))) %>% 
  ggplot(., aes(x = disease, y = city, fill = step_size_char)) +
  geom_tile() +
  geom_tile(color = "black", size = 1) +
  scale_fill_viridis_d(option = "magma") +
  coord_fixed() +
  labs(x = "", y = "") -> p4

theme_here <- theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 14),
        title = element_text(size = 14),
        legend.text = element_text(size = 12))

plot_grid(p1 + 
            labs(fill = "",
                 title = "Length of\nwithin-year cycles\n(weeks)") +
            theme_here +
            guides(fill = guide_colourbar(barwidth = 10)), 
          p2 + 
            labs(title = "Length of\nbetween-year cycles\n(weeks)", fill = "") +
            theme_here +
            guides(fill = guide_colourbar(barwidth = 10)) ,
          p3 + 
            labs(title = "Time series\ncoverage\n(days)", fill = "") +
            theme_here +
            guides(fill = guide_colourbar(barwidth = 10)), 
          p4 + 
            labs(fill = "", title = "Unit of\ntime series") +
            theme_here +
            guides(fill = guide_legend(nrow = 2)), 
          nrow = 1,
          labels = c("(A)", 
                     "(B)", 
                     "(C)", 
                     "(D)"),
          label_x = 0.2, label_y = 0.89,
          align = "hv",
          axis = "tblr") -> p_top

 ggsave("figs/fig3_v1.jpg", width = 14, height = 8)

 
get_top <- function(m = 6){
  LL_flu %>%
    map(dplyr::filter, model == "mstl") %>% 
    map(mutate, 
        ll_sum_max = max(ll_sum),
        ll_relative = if_else(ll_sum < 0, ll_sum_max/ll_sum, ll_sum/ll_sum_max),
        ll_similar = sum(ll_relative >= 0.9) - 1,
        # ll_relative = ll_sum_max/ll_sum,
        step_size1_standardised = case_when(step_size_char == "daily" ~ length_cycle1/30,
                                            step_size_char == "weekly" ~ length_cycle1/4,
                                            step_size_char == "monthly" ~ length_cycle1),
        step_size2_standardised = case_when(step_size_char == "daily" ~ length_cycle2/30,
                                            step_size_char == "weekly" ~ length_cycle2/4,
                                            step_size_char == "monthly" ~ length_cycle2),
        acf1_max = max(acf1),
        acf1_relative = acf1/acf1_max,
        acf1_similar = sum(acf1_relative >= 0.9) - 1,
        rank_ll = rank(desc(ll_sum)),
        rank_mae = rank(mae_mean),
        rank_acf = rank(acf1),
        rank_size = rank(desc(size_cycle1)),
        size_relative = size_cycle2/size_cycle1,
        rank_mean = (rank_ll + rank_acf)/2) %>% 
    map(arrange, rank_mean) %>% 
    map(~.[1:m,]) %>% 
    bind_rows() %>% 
    mutate(city = factor(city, levels = city_ordered),
           direction = if_else(city %in% c("Beijing", "Xian", "Lanzhou"),
                               "North",
                               "South"),
           disease = "Influenza") %>% 
    bind_rows(LL_rsv %>%
                map(dplyr::filter, model == "mstl") %>% 
                map(mutate, 
                    ll_sum_max = max(ll_sum),
                    ll_relative = if_else(ll_sum < 0, ll_sum_max/ll_sum, ll_sum/ll_sum_max),
                    ll_similar = sum(ll_relative >= 0.9) - 1,
                    # ll_relative = ll_sum_max/ll_sum,
                    step_size1_standardised = case_when(step_size_char == "daily" ~ length_cycle1/30,
                                                        step_size_char == "weekly" ~ length_cycle1/4,
                                                        step_size_char == "monthly" ~ length_cycle1),
                    step_size2_standardised = case_when(step_size_char == "daily" ~ length_cycle2/30,
                                                        step_size_char == "weekly" ~ length_cycle2/4,
                                                        step_size_char == "monthly" ~ length_cycle2),
                    acf1_max = max(acf1),
                    acf1_relative = acf1/acf1_max,
                    acf1_similar = sum(acf1_relative >= 0.9) - 1,
                    rank_ll = rank(desc(ll_sum)),
                    rank_mae = rank(mae_mean),
                    rank_acf = rank(acf1),
                    rank_size = rank(desc(size_cycle1)),
                    size_relative = size_cycle2/size_cycle1,
                    rank_mean = (rank_ll + rank_acf)/2) %>% 
                map(arrange, rank_mean) %>% 
                map(~.[1:m,]) %>% 
                bind_rows() %>% 
                mutate(city = factor(city, levels = city_ordered),
                       direction = if_else(city %in% c("Beijing", "Xian", "Lanzhou"),
                                           "North",
                                           "South"),
                       disease = "RSV")) -> tmp
  
  return(tmp)
}

p5_tab_5 <- get_top(m = 3) %>% mutate(top = "5%")
p5_tab_10 <- get_top(m = 6) %>% mutate(top = "10%")
p5_tab_15 <- get_top(m = 9) %>% mutate(top = "15%")

bind_rows(p5_tab_5, 
          p5_tab_10,
          p5_tab_15) %>% 
  mutate(top = factor(top, levels = c("5%", "10%", "15%"),
                      labels = c("Top 5%",
                                 "Top 10%",
                                 "Top 15%"))) -> p5_tab

p5_tab %>% 
  dplyr::filter(top == "Top 10%") %>% 
  ggplot(., aes(x = step_size2_standardised, fill = direction)) +
  geom_density(alpha = 0.5) +
  theme_bw() +
  labs(x = "Length of between-year cycles\namong top performing models",
       y = "Density",
       fill = "Geographic category",
       color = "Geographic category") +
  facet_grid(top~disease) +
  scale_fill_manual(values = c("#ffc38b", "#b63679")) +
  geom_vline(data = p5_tab %>% 
               group_by(top, disease, direction) %>% 
               dplyr::filter(top == "Top 10%") %>% 
               summarise(md = median(step_size2_standardised)),
             aes(xintercept = md,
                 color = direction),
             linetype = 2,
             size = 2) +
  scale_colour_manual(values = c("#ffc38b", "#b63679")) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),        
        strip.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14)) -> p5

ggsave("figs/fig_s1.jpg")

plot_grid(p_top, p5, nrow = 2)
ggsave("figs/fig3_v2_update.jpg", width = 14, height = 16)
       