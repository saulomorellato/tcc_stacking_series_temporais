rm(list=ls(all=TRUE))


#####  PACOTES  #####

library(tidyverse)
library(tidymodels)
library(stacks)
library(plsmod)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(timetk)
library(tictoc)



#####  CARREGAR OS DADOS  #####

df<- read.csv("electric_production.csv",header=TRUE)

df %>% glimpse()

df$DATE<- df$DATE %>% as.Date.character("%m/%d/%Y")

df %>% plot_time_series(DATE, IPG2211A2N)



##### SPLIT TRAIN/TEST/VALIDATION #####

split<- df %>% time_series_split(data_var = DATE,
                                 assess = "5 years",
                                 cumulative = TRUE)

#split<- df %>% initial_time_split(prop = 0.8)

df.train<- training(split)
df.test<- testing(split)


folds<- df.train %>% time_series_cv(data_var = DATE,
                                    assess = "1 year",
                                    skip = "1 year",
                                    cumulative = TRUE,
                                    slice_limit = 5)


folds %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(.date_var = DATE, .value = IPG2211A2N)



##### PRÉ-PROCESSAMENTO #####

# receita s/ covariáveis

rec_ts<- recipe(IPG2211A2N ~ DATE , data = df.train)


# receita c/ covariáveis

rec_reg<- recipe(IPG2211A2N ~ ., data = df.train) %>%
  step_fourier(DATE, period = 12, K = 15) %>% 
  step_date(DATE, features = "month", ordinal = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_mutate(DATE = as.numeric(DATE)) %>%
  step_mutate(DATE_sqrt = sqrt(DATE)) %>%
  step_mutate(DATE_log = log(DATE)) %>%
  step_normalize(DATE) %>% 
  step_lag(IPG2211A2N, lag = 1:5) %>% 
  step_impute_bag(all_predictors())



# receita c/ covariáveis

#recipe2<- recipe(cnt ~ ., data = df.train) %>%
#  step_fourier(dteday, period = c(7,12,30,52,90,365), K = 15) %>% 
#  step_date(dteday, features = "month", ordinal = FALSE) %>%
#  step_dummy(all_nominal_predictors()) %>% 
#  step_mutate(dteday = as.numeric(dteday)) %>%
#  step_normalize(dteday)



##### MODELOS #####

model.arima<- arima_reg() %>%
  set_engine("auto_arima") %>%
  set_mode("regression")


#model.arima<- arima_reg(non_seasonal_ar = tune(),
#                      non_seasonal_differences = tune(),
#                      non_seasonal_ma = tune(),
#                      seasonal_ar = tune(),
#                      seasonal_differences = tune(),
#                      seasonal_ma = tune()) %>%
#  set_engine("auto_arima") %>%
#  set_mode("regression")


model.arima.boost<- arima_boost(min_n = tune(),
                                trees = tune(),
                                learn_rate = tune()) %>%
  set_engine("auto_arima_xgboost") %>%
  set_mode("regression")


model.prophet.reg<- prophet_reg(changepoint_num = tune(),
                                changepoint_range = tune(),
                                growth = tune(),
                                season = tune()) %>%
  set_engine("prophet") %>%
  set_mode("regression")


model.prophet.boost<- prophet_boost(changepoint_num = tune(),
                                    changepoint_range = tune(),
                                    growth = tune(),
                                    season = tune(),
                                    min_n = tune(),
                                    trees = tune(),
                                    learn_rate = tune()) %>%
  set_engine("prophet_xgboost") %>%
  set_mode("regression")


model.pls<- parsnip::pls(num_comp = tune()) %>%
  set_engine("mixOmics") %>%
  set_mode("regression")


model.las<- linear_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")


model.rid<- linear_reg(penalty = tune(), mixture = 0) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")


model.net<- linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>%
  set_mode("regression")


model.svm.lin<- svm_linear(cost = tune(), margin = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("regression")


model.svm.pol<- svm_poly(cost = tune(), margin = tune(), degree = tune(),
                         scale_factor = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("regression")


model.svm.rbf<- svm_rbf(cost = tune(), rbf_sigma = tune(), margin = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("regression")




##### WORKFLOW #####

wf.arima<- workflow() %>%
  add_recipe(rec_ts) %>%
  add_model(model.arima)

wf.arima.boost<- workflow() %>%
  add_recipe(rec_ts) %>%
  add_model(model.arima.boost)

wf.prophet.reg<- workflow() %>%
  add_recipe(rec_ts) %>%
  add_model(model.prophet.reg)

wf.prophet.boost<- workflow() %>%
  add_recipe(rec_ts) %>%
  add_model(model.prophet.boost)

wf.pls<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.pls)

wf.las<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.las)

wf.rid<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.rid)

wf.net<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.net)

wf.svm.lin<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.svm.lin)

wf.svm.pol<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.svm.pol)

wf.svm.rbf<- workflow() %>%
  add_recipe(rec_reg) %>%
  add_model(model.svm.rbf)



##### FITTING MODELS WITHOUT HIPERPARAMETERS TUNING #####

wf.arima<- wf.arima %>% fit(df.train)



##### HIPERPARAMETERS TUNING - BAYESIAN SEARCH #####

tic()
set.seed(1)
tune.arima.boost<- tune_bayes(wf.arima.boost,
                              resamples = folds,
                              initial = 20,
                              control = control_stack_bayes(),
                              metrics = metric_set(rmse),
                              param_info = parameters(min_n(range=c(1,40)),
                                                      trees(range=c(50,2000)),
                                                      learn_rate(range=c(-100,0)))
)
toc()
# xx sec elapsed


tic()
set.seed(1)
tune.prophet.reg<- tune_bayes(wf.prophet.reg,
                              resamples = folds,
                              initial = 10,
                              control = control_stack_bayes(),
                              metrics = metric_set(rmse),
                              param_info = parameters(changepoint_num(range=c(0,50)),
                                                      changepoint_range(range=c(0.5,0.95)),
                                                      growth(values=c("linear", "logistic")),
                                                      season(values=c("additive", "multiplicative")))
)
toc()
# 83.05 sec elapsed


tic()
set.seed(1)
tune.prophet.boost<- tune_bayes(wf.prophet.boost,
                                resamples = folds,
                                initial = 10,
                                control = control_stack_bayes(),
                                metrics = metric_set(rmse),
                                param_info = parameters(changepoint_num(range=c(0,50)),
                                                        changepoint_range(range=c(0.5,0.95)),
                                                        growth(values=c("linear", "logistic")),
                                                        season(values=c("additive", "multiplicative")),
                                                        min_n(range=c(1,40)),
                                                        trees(range=c(50,2000)),
                                                        learn_rate(range=c(-100,0)))
)
toc()
# 255.15 sec elapsed


tic()
set.seed(0)
tune.pls<- tune_bayes(wf.pls,
                      resamples = folds,
                      initial = 10,
                      control = control_stack_bayes(),
                      metrics = metric_set(rmse),
                      param_info = parameters(num_comp(range=c(1,35)))
)
toc()
# 790.25 sec elapsed


tic()
set.seed(1)
tune.las<- tune_bayes(wf.las,
                      resamples = folds,
                      initial = 10,
                      control = control_stack_bayes(),
                      metrics = metric_set(rmse),
                      param_info = parameters(penalty(range=c(-100,10)))
)
toc()
# 790.57 sec elapsed


tic()
set.seed(1)
tune.rid<- tune_bayes(wf.rid,
                      resamples = folds,
                      initial = 15,
                      control = control_stack_bayes(),
                      metrics = metric_set(rmse),
                      param_info = parameters(penalty(range=c(-100,10)))
)
toc()
# 786.59 sec elapsed



tic()
set.seed(1)
tune.net<- tune_bayes(wf.net,
                      resamples = folds,
                      initial = 10,
                      control = control_stack_bayes(),
                      metrics = metric_set(rmse),
                      param_info = parameters(penalty(range=c(-100,10)),
                                              mixture(range=c(0,1)))
)
toc()
# 803.75 sec elapsed


tic()
set.seed(0)
tune.svm.lin<- tune_bayes(wf.svm.lin,
                          resamples = folds,
                          initial = 10,
                          control = control_stack_bayes(),
                          metrics = metric_set(rmse),
                          param_info = parameters(cost(range=c(-10,5)),
                                                  svm_margin(range=c(0,0.5)))
)
toc()
# 894.89 sec elapsed


tic()
set.seed(0)
tune.svm.pol<- tune_bayes(wf.svm.pol,
                          resamples = folds,
                          initial = 10,
                          control = control_stack_bayes(),
                          metrics = metric_set(rmse),
                          param_info = parameters(cost(range=c(-10,5)),
                                                  degree(range=c(1,3)),
                                                  scale_factor(range=c(-10,-1)),
                                                  svm_margin(range=c(0,0.5)))
)
toc()
# 861.52 sec elapsed


tic()
set.seed(0)
tune.svm.rbf<- tune_bayes(wf.svm.rbf,
                          resamples = folds,
                          initial = 10,
                          control = control_stack_bayes(),
                          metrics = metric_set(rmse),
                          param_info = parameters(cost(range=c(-10,5)),
                                                  rbf_sigma(range=c(-10,0)),
                                                  svm_margin(range=c(0,0.5)))
)
toc()
# 833.48 sec elapsed



## ESCOLHENDO O MELHOR (BEST RMSE)

#show_best(tune.arima,n=3)
show_best(tune.arima.boost,n=3)
show_best(tune.prophet.reg,n=3)
show_best(tune.prophet.boost,n=3)
show_best(tune.pls,n=3)
show_best(tune.las,n=3)
show_best(tune.rid,n=3)
show_best(tune.net,n=3)
show_best(tune.svm.lin,n=3)
show_best(tune.svm.pol,n=3)
show_best(tune.svm.rbf,n=3)



##### TUNED WORKFLOW #####

wf.arima.boost<- wf.arima.boost %>% 
  finalize_workflow(select_best(tune.arima.boost)) %>% 
  fit(df.train)

wf.prophet.reg<- wf.prophet.reg %>% 
  finalize_workflow(select_best(tune.prophet.reg)) %>% 
  fit(df.train)

wf.prophet.boost<- wf.prophet.boost %>% 
  finalize_workflow(select_best(tune.prophet.boost)) %>% 
  fit(df.train)

wf.pls<- wf.pls %>% 
  finalize_workflow(select_best(tune.pls)) %>% 
  fit(df.train)

wf.las<- wf.las %>% 
  finalize_workflow(select_best(tune.las)) %>% 
  fit(df.train)

wf.rid<- wf.rid %>% 
  finalize_workflow(select_best(tune.rid)) %>% 
  fit(df.train)

wf.net<- wf.net %>% 
  finalize_workflow(select_best(tune.net)) %>% 
  fit(df.train)

wf.svm.lin<- wf.svm.lin %>% 
  finalize_workflow(select_best(tune.svm.lin)) %>% 
  fit(df.train)

wf.svm.pol<- wf.svm.pol %>% 
  finalize_workflow(select_best(tune.svm.pol)) %>% 
  fit(df.train)

wf.svm.rbf<- wf.svm.rbf %>% 
  finalize_workflow(select_best(tune.svm.rbf)) %>% 
  fit(df.train)



### LISTA MODELOS AND ENSEMBLE ###

models_tab<- modeltime_table(wf.arima,
                             wf.arima.boost,
                             wf.prophet.reg,
                             wf.prophet.boost,
                             wf.pls,
                             wf.las,
                             wf.rid,
                             wf.net,
                             wf.svm.lin,
                             wf.svm.pol,
                             wf.svm.rbf)

# SIMPLE ENSEMBLE

ensemble.mean<- models_tab %>% ensemble_average(type="mean")
ensemble.median<- models_tab %>% ensemble_average(type="median")


# COMPLEX ENSEMBLE

tic()
submodel_predictions <- models_tab %>%
  modeltime_fit_resamples(resamples = split,
                          control = control_resamples(verbose = TRUE))
toc()


ensemble_las <- submodel_predictions %>%
  ensemble_model_spec(model_spec = linear_reg(penalty = tune(), mixture = 1) %>% 
                        set_engine("glmnet"),
                      grid = 2,
                      control = control_grid(verbose = TRUE))


ensemble_rid <- submodel_predictions %>%
  ensemble_model_spec(model_spec = linear_reg(penalty = tune(), mixture = 0) %>% 
                        set_engine("glmnet"),
                      grid = 2,
                      control = control_grid(verbose = TRUE))


ensemble_net <- submodel_predictions %>%
  ensemble_model_spec(model_spec = linear_reg(penalty = tune(), mixture = tune()) %>% 
                        set_engine("glmnet"),
                      grid = 2,
                      control = control_grid(verbose = TRUE))


ensemble_pls <- submodel_predictions %>%
  ensemble_model_spec(model_spec = parsnip::pls(num_comp = tune()) %>%
                        set_engine("mixOmics"),
                      grid = 2,
                      control = control_grid(verbose = TRUE))




### DESEMPENHO DOS MODELOS ###

# GRÁFICO

modeltime_table(wf.arima,
                wf.arima.boost,
                wf.prophet.reg,
                wf.prophet.boost,
                wf.pls,
                wf.las,
                wf.rid,
                wf.net,
                wf.svm.lin,
                wf.svm.pol,
                wf.svm.rbf) %>% 
  modeltime_calibrate(new_data = df.test) %>% 
  modeltime_forecast(new_data = df.test, 
                     actual_data = df) %>% 
  plot_modeltime_forecast()


#TABELA

medidas<- modeltime_table(wf.arima,
                          wf.arima.boost,
                          wf.prophet.reg,
                          wf.prophet.boost,
                          wf.pls,
                          wf.las,
                          wf.rid,
                          wf.net,
                          wf.svm.lin,
                          wf.svm.pol,
                          wf.svm.rbf) %>% 
  modeltime_calibrate(new_data = df.test) %>% 
  modeltime_accuracy()

medidas

# escolher o "melhor" modelo segundo qual medida?
#best<- order(medidas$rmse)[1]
#best<- order(medidas$mae)[1]
#best<- order(medidas$mape)[1]
#best<- order(medidas$rsq)[1]

# escolher o "melhor" modelo segundo um desempenho geral
best<- order((rank(medidas$mae)+rank(medidas$mape)+rank(medidas$rmse)+rank(-medidas$rsq))/4)[1]

if(best==1){wf.best<-wf.arima}
if(best==2){wf.best<-wf.arima.boost}
if(best==3){wf.best<-wf.prophet.reg}
if(best==4){wf.best<-wf.prophet.boost}
if(best==5){wf.best<-wf.pls}
if(best==6){wf.best<-wf.las}
if(best==7){wf.best<-wf.rid}
if(best==8){wf.best<-wf.net}
if(best==9){wf.best<-wf.svm.lin}
if(best==10){wf.best<-wf.svm.pol}
if(best==11){wf.best<-wf.svm.rbf}

wf.best



### FINALIZANDO O MODELO ###

wf.final<- fit(wf.best, df)



### SALVANDO O MODELO FINAL ###

saveRDS(wf.final,"wf_electric_production.rds")




