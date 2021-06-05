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


holdout_set <- read_csv("data/test.csv",
                        col_types = cols(
                          .default = col_character(),
                          game_id = col_double(),
                          min_players = col_double(),
                          max_players = col_double(),
                          avg_time = col_double(),
                          min_time = col_double(),
                          max_time = col_double(),
                          year = col_double(),
                         # geek_rating = col_double(),
                          num_votes = col_double(),
                          age = col_double(),
                          owned = col_double(),
                          category9 = col_character(),
                          category10 = col_character(),
                          category11 = col_character(),
                          category12 = col_character()
                        )) %>% 
  select(-c(mechanic, designer, names, contains("category")), category1, category2)



clean_df <- raw_df %>% 
  mutate(max_players = ifelse(max_players == 99, min_players, max_players),
         # don't be an idiot, step_impute()!!!
         year = ifelse(year < 1000, NA, year)) %>%
  filter(max_time < 5000) %>% 
  select(-c(game_id, mechanic, designer, names, contains("category")), category1, category2)
  


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


#" Don't even use designer, way too many of them!!!!!

# Only use category 1
# if I have time use textrecipes on mechanic (i don't remember nlp right now!!!!)


# Partition ---------------------------------------------------------------
# THIS IS WHERE THE MAGIC BEGINS BABY!!!!

init_split <- initial_split(clean_df)

train_df <- training(init_split)
test_df <- testing(init_split)

folds <- vfold_cv(train_df)


# RECIPES - CHEF IS ENTERING THE KITCHEN ----------------------------------


base_rec <- recipe(geek_rating ~ min_players + 
                     max_players + avg_time + min_time + max_time  + 
         num_votes, data = train_df) 

int_rec <- recipe(geek_rating ~ min_players + 
         max_players + avg_time + min_time + max_time +
         year + 
         num_votes + owned + age + category1, data = train_df) %>% 
  step_impute_knn(year) %>% 
  step_other(category1) %>% 
  step_dummy(category1) %>% 
  step_range(owned, num_votes) %>% 
  step_pca(contains("time")) %>% 
  step_other(age) %>% 
  step_dummy(age) %>% 
  step_naomit(all_numeric_predictors())



adv_rec <- recipe(geek_rating ~ min_players + 
         max_players + avg_time + min_time + max_time +
         year + 
         num_votes + owned + age + category1 + category2,
         data = train_df) %>% 
  step_impute_knn(year) %>% 
  step_other(category1, threshold = 0.03) %>% 
  step_other(category2, threshold = 0.01) %>% 
  step_unknown(category2) %>% 
  step_dummy(category1, category2) %>% 
  step_range(owned, num_votes) %>% 
  step_pca(contains("time")) %>% 
  step_other(age) %>% 
  step_dummy(age) %>% 
  step_naomit(all_numeric_predictors())


last_rec <- 
  recipe(geek_rating ~ min_players + 
           max_players + avg_time + min_time + max_time +
           year + 
           num_votes + owned + age + category1 + category2,
         data = train_df) %>% 
  step_impute_knn(year) %>% 
  step_other(category1) %>% 
  step_other(category2) %>% 
  step_unknown(category2) %>% 
  step_dummy(category1, category2) %>% 
  step_range(owned, num_votes) %>% 
  step_pca(contains("time")) %>% 
  step_other(age) %>% 
  step_dummy(age) %>% 
  step_naomit(all_numeric_predictors())




recipes <- list(int = int_rec, base = base_rec,
                adv = adv_rec)

# model specifications  ---------------------------------------------------


xgb_spec <-
  boost_tree(tree_depth = tune(), trees = tune(),
             learn_rate = tune(), min_n = tune(), 
             loss_reduction = tune(), sample_size = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')

enet_spec <-
  linear_reg(penalty = tune(), 
             mixture = tune()) %>%
  set_engine('glmnet')

lm_spec <-
  linear_reg() %>%
  set_engine('lm')

nnet_spec <-
  mlp(hidden_units = tune(), 
      penalty = tune(),
      epochs = tune()) %>%
  set_engine('nnet') %>%
  set_mode('regression')

svm_spec <-
  svm_rbf(cost = tune(), rbf_sigma = tune(), margin = tune()) %>%
  set_engine('kernlab') %>%
  set_mode('regression')


rf_spec <-
  rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine('ranger') %>%
  set_mode('regression')




models <- list(enet = enet_spec,
               nnet = nnet_spec,
               xgb = xgb_spec,
               svg = svm_spec,
               rf = rf_spec)


# workflowsets BABY -------------------------------------------------------


wfs <- workflow_set(recipes, models, cross = TRUE)

fits <- workflow_map(wfs, resamples = folds, verbose = TRUE,
             control = control_grid(parallel_over = "everything"))


# model evaluation --------------------------------------------------------

top_15_current_models <- fits %>% 
  unnest(result) %>% 
  unnest(.metrics) %>% 
  filter(.metric == "rmse") %>% 
  slice_min(.estimate, n = 15) 
  


# Preliminary model -------------------------------------------------------


xgb_controls <- top_15_current_models %>% 
  filter(str_detect(wflow_id, 'xgb')) %>% 
  dplyr::slice(1) %>% 
  select(where(~!is.na(.))) %>% 
  select(trees, min_n, tree_depth, learn_rate,
         sample_size, loss_reduction)

xgb_wf <- pull_workflow(fits, "int_xgb")

xgb_fit <- xgb_wf %>% 
  finalize_workflow(xgb_controls) %>% 
  fit(train_df)

init_xgb_preds <- augment(xgb_fit, holdout_set) %>% 
  select(game_id, geek_rating = .pred)

#write_csv(init_xgb_preds, "preds/init_xgb_preds.csv")

predict(xgb_fit, holdout_set) %>% 
  ggplot(aes(.pred)) +
  geom_histogram()


#write_rds(fits, "models/last_models.rds")

# final models ------------------------------------------------------------
final_mod_tunes <- fits %>% 
  unnest(result) %>% 
  unnest(.metrics) %>% 
  filter(.metric == "rmse") %>% 
  arrange(.estimate) %>% 
  group_by(wflow_id) %>% 
  slice_min(.estimate, n = 1) 

# elastic net
enet_tunes <- final_mod_tunes %>% 
  ungroup() %>%  
  dplyr::slice(1) %>% 
  select(penalty, mixture)

final_enet <- pull_workflow(fits, "adv_enet") %>% 
  finalize_workflow(enet_tunes) %>% 
  fit(train_df)


augment(final_enet, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/enet-preds.csv")

# final nnet
nnet_control <- final_mod_tunes %>% 
  ungroup() %>%  
  slice(2) %>% 
  select(hidden_units, 
         penalty, epochs)


final_nnet <- pull_workflow(fits, "adv_nnet") %>% 
  finalize_workflow(nnet_control) %>% 
  fit(train_df)


augment(final_nnet, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/nnet-preds.csv")

# Random forest

rf_control <- final_mod_tunes %>% 
  ungroup() %>%  
  slice(3) %>% 
  select(mtry, min_n)


final_rf <- pull_workflow(fits, "adv_rf") %>% 
  finalize_workflow(rf_control) %>% 
  fit(train_df)

augment(final_rf, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/rf-preds.csv")

# svm

svm_control <- final_mod_tunes %>% 
  ungroup() %>%  
  slice(4) %>% 
  select(cost, rbf_sigma, margin)

final_svm <-  pull_workflow(fits, "recipe_svg") %>% 
  finalize_workflow(svm_control) %>% 
  fit(train_df)

augment(final_svm, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/svm-preds.csv")


# xgb

xgb_controls <- final_mod_tunes %>% 
  ungroup() %>%  
  slice(5) %>% 
  select(tree_depth, trees,
         learn_rate, min_n, 
         loss_reduction, sample_size)
  
final_xgb <- pull_workflow(fits, "adv_xgb") %>% 
  finalize_workflow(xgb_controls) %>% 
  fit(train_df)


augment(final_xgb, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/xgb-preds.csv")


# last ditch effort -------------------------------------------------------

# last recipe

last_rec <- 
  recipe(geek_rating ~ min_players + 
           max_players + avg_time + min_time + max_time +
           year + 
           num_votes + owned + age + category1 + category2,
         data = train_df) %>% 
  step_impute_knn(year) %>% 
  step_other(category1) %>% 
  step_other(category2) %>% 
  step_unknown(category2) %>% 
  step_dummy(category1, category2) %>% 
  step_range(owned, num_votes) %>% 
  step_pca(contains("time")) %>% 
  step_other(age) %>% 
  step_dummy(age) %>% 
  step_naomit(all_numeric_predictors())

last_models <- list(xgb = xgb_spec, rf = rf_spec)
last_recipe <- list(sob = last_rec)

lastwf <- workflow_set(last_recipe, last_models)

last_fits <- workflow_map(lastwf, resamples = folds, verbose = TRUE,
             control = control_grid(parallel_over = "everything"))



last_controls <- last_fits %>% 
  unnest(result) %>% 
  unnest(.metrics) %>% 
  filter(.metric == "rmse") %>% 
  arrange(.estimate) %>% 
  group_by(wflow_id) %>% 
  slice_min(.estimate, n = 1) 

# last xgb
last_xgb_controls <- last_controls %>% 
  ungroup() %>% 
  slice(2) %>% 
  select(tree_depth, trees,
       learn_rate, min_n, 
       loss_reduction, sample_size)

last_xgb <- pull_workflow(last_fits, "sob_xgb") %>% 
  finalize_workflow(last_xgb_controls) %>% 
  fit(train_df)

augment(last_xgb, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/last-xgb-preds.csv")

# last rf
last_rf_controls <- last_controls %>% 
  ungroup() %>% 
  slice(1) %>% 
  select(mtry, min_n)

last_rf <- pull_workflow(last_fits, "sob_rf") %>% 
  finalize_workflow(last_rf_controls) %>% 
  fit(train_df)

augment(last_rf, holdout_set) %>% 
  select(game_id, geek_rating = .pred) %>% 
  write_csv("preds/last-rf-preds.csv")


