rm(list=ls(all=TRUE))


#####  PACOTES  #####

library(tidyverse)
library(tidymodels)
library(stacks)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(timetk)
library(tictoc)



#####  CARREGAR OS DADOS  #####

df<- read.csv("G:/Meu Drive/estatistica/modeltime/electric_production.csv",header=TRUE)

df %>% glimpse()

df %>% plot_time_series(DATE, IPG2211A2N)



##### SPLIT TRAIN/TEST/VALIDATION #####

split<- df %>% time_series_split(data_var = dteday,
                                 assess = "3 months",
                                 cumulative = TRUE)

#split<- df %>% initial_time_split(prop = 0.8)

df.train<- training(split)
df.test<- testing(split)


folds<- df.train %>% time_series_cv(data_var = dteday,
                                    assess = "2 months",
                                    skip = "2 months",
                                    cumulative = TRUE,
                                    slice_limit = 5)


folds %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(.date_var = dteday, .value = cnt)



##### PRÉ-PROCESSAMENTO #####

# receita s/ covariáveis

recipe0<- recipe(cnt ~ dteday , data = df.train)


# receita c/ covariáveis

recipe1<- recipe(cnt ~ ., data = df.train) %>%
  step_fourier(dteday, period = c(7,365), K = 15) %>% 
  step_date(dteday, features = "month", ordinal = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_mutate(dteday = as.numeric(dteday)) %>%
  step_normalize(dteday)


# receita c/ covariáveis

recipe2<- recipe(cnt ~ ., data = df.train) %>%
  step_fourier(dteday, period = c(7,12,30,52,90,365), K = 15) %>% 
  step_date(dteday, features = "month", ordinal = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_mutate(dteday = as.numeric(dteday)) %>%
  step_normalize(dteday)



##### MODELOS #####

fit.prophet<- prophet_reg(changepoint_num = tune(),
                          changepoint_range = tune()) %>%
  set_engine("prophet") %>%
  set_mode("regression")

fit.arima<- arima_reg(non_seasonal_ar = tune(),
                      non_seasonal_differences = tune(),
                      non_seasonal_ma = tune(),
                      seasonal_ar = tune(),
                      seasonal_differences = tune(),
                      seasonal_ma = tune()) %>%
  set_engine("auto_arima") %>%
  set_mode("regression")

fit.arima.boost<- arima_boost(learn_rate = tune(),
                              trees = tune()) %>%
  set_engine("auto_arima_xgboost") %>%
  set_mode("regression")

fit.las<- linear_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")

fit.rid<- linear_reg(penalty = tune(), mixture = 0) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")

fit.net<- linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")

fit.poi<- poisson_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")



##### WORKFLOW #####

wf.prophet<- workflow() %>%
  add_recipe(recipe0) %>%
  add_model(fit.prophet)

wf.arima<- workflow() %>%
  add_recipe(recipe0) %>%
  add_model(fit.arima)

wf.arima.boost<- workflow() %>%
  add_recipe(recipe0) %>%
  add_model(fit.arima.boost)

wf.las1<- workflow() %>%
  add_recipe(recipe1) %>%
  add_model(fit.las)

wf.rid1<- workflow() %>%
  add_recipe(recipe1) %>%
  add_model(fit.rid)

wf.net1<- workflow() %>%
  add_recipe(recipe1) %>%
  add_model(fit.net)

wf.las2<- workflow() %>%
  add_recipe(recipe2) %>%
  add_model(fit.las)

wf.rid2<- workflow() %>%
  add_recipe(recipe2) %>%
  add_model(fit.rid)

wf.net2<- workflow() %>%
  add_recipe(recipe2) %>%
  add_model(fit.net)

wf.poi1<- workflow() %>%
  add_recipe(recipe1) %>%
  add_model(fit.poi)

wf.poi2<- workflow() %>%
  add_recipe(recipe2) %>%
  add_model(fit.poi)


##### HIPERPARAMETERS TUNING - BAYESIAN SEARCH #####

tic()
set.seed(1)
tune.prophet<- tune_bayes(wf.prophet,
                          resamples = folds,
                          initial = 10,
                          control = control_stack_bayes(),
                          metrics = metric_set(rmse),
                          param_info = parameters(changepoint_num(range=c(0,50)),
                                                  changepoint_range(range=c(0.5,0.95)))
)
toc()
# 49.11 sec elapsed



tic()
set.seed(1)
tune.arima<- tune_bayes(wf.arima,
                        resamples = folds,
                        initial = 10,
                        control = control_stack_bayes(),
                        metrics = metric_set(rmse),
                        param_info = parameters(non_seasonal_ar(range=c(0,7)),
                                                non_seasonal_differences(range=c(1,2)),
                                                non_seasonal_ma(range=c(0,7)),
                                                seasonal_ar(range=c(0,2)),
                                                seasonal_differences(range=c(0,1)),
                                                seasonal_ma(range=c(0,2)))
)
toc()


tic()
set.seed(1)
tune.arima.boost<- tune_bayes(wf.arima.boost,
                              resamples = folds,
                              initial = 15,
                              control = control_stack_bayes(),
                              metrics = metric_set(rmse),
                              param_info = parameters(learn_rate(range=c(-100,0)),
                                                      trees(range=c(0,2000)))
)
toc()
# 148.28 sec elapsed


tic()
set.seed(1)
tune.las1<- tune_bayes(wf.las1,
                       resamples = folds,
                       initial = 10,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)))
)
toc()
# 32.54 sec elapsed



tic()
set.seed(1)
tune.rid1<- tune_bayes(wf.rid1,
                       resamples = folds,
                       initial = 15,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)))
)
toc()
# 32.53 sec elapsed



tic()
set.seed(1)
tune.net1<- tune_bayes(wf.net1,
                       resamples = folds,
                       initial = 10,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)),
                                               mixture(range=c(0,1)))
)
toc()
# 42.04 sec elapsed


tic()
set.seed(1)
tune.las2<- tune_bayes(wf.las2,
                       resamples = folds,
                       initial = 10,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)))
)
toc()
# 32.54 sec elapsed



tic()
set.seed(1)
tune.rid2<- tune_bayes(wf.rid2,
                       resamples = folds,
                       initial = 15,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)))
)
toc()
# 32.53 sec elapsed



tic()
set.seed(1)
tune.net2<- tune_bayes(wf.net2,
                       resamples = folds,
                       initial = 10,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)),
                                               mixture(range=c(0,1)))
)
toc()
# 42.04 sec elapsed


tic()
set.seed(1)
tune.poi1<- tune_bayes(wf.poi1,
                       resamples = folds,
                       initial = 10,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)),
                                               mixture(range=c(0,1)))
)
toc()
# 42.04 sec elapsed


tic()
set.seed(1)
tune.poi2<- tune_bayes(wf.poi2,
                       resamples = folds,
                       initial = 10,
                       control = control_stack_bayes(),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-100,10)),
                                               mixture(range=c(0,1)))
)
toc()
# 42.04 sec elapsed



## ESCOLHENDO O MELHOR (BEST RMSE)

show_best(tune.prophet,n=3)
show_best(tune.arima,n=3)
show_best(tune.arima.boost,n=3)
show_best(tune.las1,n=3)
show_best(tune.rid1,n=3)
show_best(tune.net1,n=3)
show_best(tune.las2,n=3)
show_best(tune.rid2,n=3)
show_best(tune.net2,n=3)
show_best(tune.poi1,n=3)
show_best(tune.poi2,n=3)


##### TUNED WORKFLOW #####

wf.prophet<- wf.prophet %>% 
  finalize_workflow(select_best(tune.prophet)) %>% 
  fit(df.train)

wf.arima<- wf.arima %>% 
  finalize_workflow(select_best(tune.arima)) %>% 
  fit(df.train)

wf.arima.boost<- wf.arima.boost %>% 
  finalize_workflow(select_best(tune.arima.boost)) %>% 
  fit(df.train)

wf.las1<- wf.las1 %>% 
  finalize_workflow(select_best(tune.las1)) %>% 
  fit(df.train)

wf.rid1<- wf.rid1 %>% 
  finalize_workflow(select_best(tune.rid1)) %>% 
  fit(df.train)

wf.net1<- wf.net1 %>% 
  finalize_workflow(select_best(tune.net1)) %>% 
  fit(df.train)

wf.las2<- wf.las2 %>% 
  finalize_workflow(select_best(tune.las2)) %>% 
  fit(df.train)

wf.rid2<- wf.rid2 %>% 
  finalize_workflow(select_best(tune.rid2)) %>% 
  fit(df.train)

wf.net2<- wf.net2 %>% 
  finalize_workflow(select_best(tune.net2)) %>% 
  fit(df.train)

wf.poi1<- wf.poi1 %>% 
  finalize_workflow(select_best(tune.poi1)) %>% 
  fit(df.train)

wf.poi2<- wf.poi2 %>% 
  finalize_workflow(select_best(tune.poi2)) %>% 
  fit(df.train)



#TABELA

modeltime_table(wf.prophet,
                wf.arima,
                wf.arima.boost,
                wf.las1,
                wf.rid1,
                wf.net1,
                wf.las2,
                wf.rid2,
                wf.net2,
                wf.poi1,
                wf.poi2) %>% 
  modeltime_calibrate(new_data = df.test) %>% 
  modeltime_accuracy()


# GRÁFICO

modeltime_table(wf.prophet,
                wf.arima,
                wf.arima.boost,
                wf.las1,
                wf.rid1,
                wf.net1,
                wf.las2,
                wf.rid2,
                wf.net2,
                wf.poi1,
                wf.poi2) %>% 
  modeltime_calibrate(new_data = df.test) %>% 
  modeltime_forecast(new_data = df.test, 
                     actual_data = df) %>% 
  plot_modeltime_forecast()


wf.best<- wf.prophet
wf.best  # workflow visualization

wf.train<- fit(wf.best, df.train)

