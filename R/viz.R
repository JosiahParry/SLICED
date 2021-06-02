library(tidytext)
library(tidyverse)
library(tidymodels)


tidymodels_prefer()
doParallel::registerDoParallel(cores = 4)

theme_set(theme_light())

# Data import -------------------------------------------------------------
raw_df <- read_csv("data/train.csv",
                   col_types = cols(
                     .default = col_character(),
                     game_id = col_double(),
                     min_players = col_double(),
                     max_players = col_double(),
                     avg_time = col_double(),
                     min_time = col_double(),
                     max_time = col_double(),
                     year = col_double(),
                     geek_rating = col_double(),
                     num_votes = col_double(),
                     age = col_double(),
                     owned = col_double(),
                     category9 = col_character(),
                     category10 = col_character(),
                     category11 = col_character(),
                     category12 = col_character()
                   ))


clean_df <- raw_df %>% 
  mutate(max_players = ifelse(max_players == 99, min_players, max_players),
         # don't be an idiot, step_impute()!!!
         year = ifelse(year < 1000, NA, year)) %>%
  filter(max_time < 5000) %>% 
  select(-c(game_id, mechanic, designer, names, contains("category")), category1)



ggplot(clean_df, aes(max_time)) +
  geom_histogram() +
  labs(title = "Some seriously long games I won't play",
       x = "Game Duration",
       y = "Frequency")

clean_df %>% 
  mutate(category = fct_lump(category1, prop = 0.05)) %>% 
  ggplot(aes(min_time, max_time, size = max_players)) +
  geom_point(alpha = 0.12) +
  facet_wrap("category",
             scales = "free") +
  labs(title = "Min / Max time by game category")



clean_df


raw_df %>% 
  unnest_tokens(word, mechanic) %>% 
  count(word) %>% 
  slice_max(n, n = 15) %>% 
  mutate(word = fct_reorder(word, n )) %>% 
  ggplot(aes(n, word)) +
  geom_col() + 
  labs(title = "Most Frequnt Game Mechanics",
       caption = "though I don't think you\n need to be able to work on a car")
  

raw_df %>% 
  mutate(category1 = fct_lump(category1, prop = 0.05)) %>% 
  unnest_tokens(word, mechanic) %>% 
  dplyr::count(category1, word) %>% 
  bind_tf_idf(word, category1, n) %>% 
  group_by(category1) %>% 
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>% 
  mutate(word = fct_reorder(word, tf_idf)) %>% 
  ggplot(aes(tf_idf, word)) +
  geom_col() + 
  facet_wrap("category1", scales = "free_y") +
  labs(title = "Most unique mechanic by category")


raw_df %>% 
  mutate(category = fct_lump(category1, prop = 0.05)) %>% 
  ggplot(aes(geek_rating, category)) +
  ggridges::geom_density_ridges()
  
  



