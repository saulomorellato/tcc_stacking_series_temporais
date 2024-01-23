rm(list=ls(all=TRUE))


#####  PACOTES  #####

library(tidyverse)
library(tidymodels)
library(stacks)
library(plsmod)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(modeltime.h2o)
library(timetk)
library(tictoc)

h2o.init()



#####  CARREGAR OS DADOS  #####

df<- read.csv("electric_production.csv", header=TRUE)

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

rec<- recipe(IPG2211A2N ~ DATE , data = df.train)


# receita c/ covariáveis

rec<- recipe(IPG2211A2N ~ ., data = df.train) %>%
  step_fourier(DATE, period = 12, K = 15) %>% 
  step_date(DATE, features = "month", ordinal = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_mutate(DATE2 = as.numeric(DATE)) %>%
  step_mutate(DATE_sqrt = sqrt(DATE2)) %>%
  step_mutate(DATE_log = log(DATE2)) %>%
  step_normalize(DATE2) %>% 
  step_lag(IPG2211A2N, lag = 1:5) %>% 
  step_naomit()
  #step_impute_bag(all_predictors())



##### MODELOS #####

model.aut<- automl_reg() %>%
  set_engine(engine = "h2o",
             max_runtime_secs = 3600, 
             max_runtime_secs_per_model = 1000,
             nfolds = 5,
             max_models = 1000,
             exclude_algos = c("DeepLearning"),
             verbosity = NULL,
             seed = 0) %>%
  set_mode("regression")



##### WORKFLOW #####

wf.aut<- workflow() %>%
  add_recipe(rec) %>%
  add_model(model.aut)



##### FITTING MODELS AND CHOOSING THE BEST #####

tic()
wf.fitted<- fit(wf.aut,df.train)
toc()
# 531.72 sec elapsed
# 3792.1 sec elapsed



##### VISUALIZATION #####

wf.fitted
automl_leaderboard(wf.fitted)



### UPDATE (OPTIONAL) ###

#automl_update_model(wf.fitted, model_id = "StackedEnsemble_AllModels_AutoML_20210319_204825")



### DESEMPENHO DOS MODELOS ###

# GRÁFICO

modeltime_table(wf.fitted) %>% 
  modeltime_calibrate(new_data = df.test) %>% 
  modeltime_forecast(new_data = df.test, 
                     actual_data = df) %>% 
  plot_modeltime_forecast()


#TABELA

medidas<- modeltime_table(wf.fitted) %>% 
  modeltime_calibrate(new_data = df.test) %>% 
  modeltime_accuracy()

medidas




### FINALIZANDO O MODELO ###

wf.final<- fit(wf.fitted, df)



### SALVANDO O MODELO FINAL ###

saveRDS(wf.final,"wf_auto_electric_production.rds")
#save_h2o_model(wf.final,"wf_auto_electric_production2")


### ENCERRAR SESSÃO ###

h2o.shutdown(prompt = FALSE)

