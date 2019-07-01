rm(list=ls())
library(data.table)  
library(tidyverse)
library(caret)
library("fastDummies")
library(h2o)

cat("reading the train and test data (with fread from data.table) \n")
df2 <- data.table::fread("C:\\Users\\python\\base_SISVAN_CAD.csv",sep = "auto", header= TRUE, 
                         na.strings = "NA") %>% data.frame()

#df2 <- data.table::fread("/Dados/base_completa.csv",sep = "auto", header= TRUE, 
#                         na.strings = "NA") %>% data.frame()

var_select <- c("pesocat","idademae", 
                "escmae", "consultas", "sexo", "peso", "seriescmae", 
                "racacormae", "qtdgestant", "qtdpartnor", "qtdpartces", 
                "semagestac", "consprenat", "mesprenat", "stcesparto", 
                "tpnascassi", "tprobson", "racacor", "cod_raca_cor_pessoa_eq", 
                "cod_local_domic_fam_eq", "qtd_comodos_domic_fam_eq", 
                "qtd_pessoas_domic_fam_eq", "qtd_pessoas_fam_count", 
                "cd_grau_instrucao_v6", "premat", "paridade", 
                "dens_domic_cat", "cod_abaste_agua_cat", "cod_escoa_sanitario_cat", 
                "cod_destino_lixo_domic_cat", "qtd_pessoas", "IDHM", "IDHM_E", 
                "IDHM_L", "IDHM_R", "UF", "regiao", "meses_2004", "meses_2005", 
                "meses_2006", "meses_2007", "meses_2008", "meses_2009", 
                "meses_2010",  "meses_2011", "meses_2012", "meses_2013", "meses_2014", 
                "meses_2015",  "total_meses_bf",  "bf_tercil", "nasc_res")

df3 <- df2[,var_select]
rm(df2); rm(var_select)

#== Create Partition ==#
ind <- createDataPartition(df3$pesocat, p = .85, list = FALSE)
training <- df3[ind,]
testing <- df3[-ind,]
rm(df3)


dammy_var <- c("escmae", "consultas", "racacormae", "stcesparto", "tpnascassi", "regiao", "bf_tercil")

normalize_var <- c("idademae", "qtdgestant", "qtdpartnor", "qtdpartces", "semagestac", "consprenat", "mesprenat", "tprobson", "qtd_comodos_domic_fam_eq", 
                   "qtd_pessoas_domic_fam_eq", "qtd_pessoas_fam_count", "paridade", "total_meses_bf")

testing2 <- testing[,c("pesocat", "premat", "cod_abaste_agua_cat", "cod_escoa_sanitario_cat", "cod_destino_lixo_domic_cat", "nasc_res",
                       "sexo", "cod_local_domic_fam_eq", dammy_var,normalize_var)] %>% data.frame()
testing2 <- testing2 %>% mutate(sexo = if_else(sexo == 1, 0, 1))
testing2 <- testing2 %>% mutate(cod_local_domic_fam_eq = if_else(cod_local_domic_fam_eq == 1, 0, 1))

training2 <- training[,c("pesocat", "premat", "cod_abaste_agua_cat", "cod_escoa_sanitario_cat", "cod_destino_lixo_domic_cat", "nasc_res",
                         "sexo", "cod_local_domic_fam_eq", dammy_var,normalize_var)] %>% data.frame()
training2 <- training2 %>% mutate(sexo = if_else(sexo == 1, 0, 1))
training2 <- training2 %>% mutate(cod_local_domic_fam_eq = if_else(cod_local_domic_fam_eq == 1, 0, 1))

rm(testing)
rm(training)
rm(ind)


# Create dummy variable

training2 <- fastDummies::dummy_cols(training2, select_columns = dammy_var)
testing2 <- fastDummies::dummy_cols(testing2, select_columns = dammy_var)



# normalized
normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

for (j in normalize_var) {
  training2[[j]] <- normalize(training2[[j]])
}


training2 <- training2[,!(names(training2) %in% dammy_var)]
data.table::fwrite(training2, file = "C:\\Users\\treino.csv",quote = "auto",sep = ";",dec = ".", row.names = F, col.names = T)

testing2 <- testing2[,!(names(testing2) %in% dammy_var)]
data.table::fwrite(testing2, file = "C:\\Users\\teste.csv",quote = "auto",sep = ";",dec = ".", row.names = F, col.names = T)


treino <- data.table::fread("C:\\Users\\treino.csv",sep = "auto", header= TRUE, 
                            na.strings = "NA") %>% data.frame()

teste <- data.table::fread("C:\\Users\\teste.csv",sep = "auto", header= TRUE, 
                           na.strings = "NA") %>% data.frame()



###############               ###############
# ============ Deep Learning   ============ #
###############               ###############

path = "C:\\Users\\python\\"
setwd(path)

setDT(treino) #convert the object to data.table
setDT(teste)

#check target variable
treino[,.N/nrow(treino),premat] #binary in nature check if data is imbalanced
teste[,.N/nrow(teste),pesocat] #binary in nature check if data is imbalanced

#https://www.hackerearth.com/blog/machine-learning/understanding-deep-learning-parameter-tuning-with-mxnet-h2o-package-in-r/
#https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning

## Start cluster with all available threads
h2o.init(nthreads = -1)
#localH2o <- h2o.init(nthreads = -1, max_mem_size = "20G")

#load data on H2o
trainh2o <- as.h2o(treino)
testh2o <- as.h2o(teste)

#set variables
y <- "pesocat"
x <- setdiff(colnames(trainh2o),y)

#set parameter space

activation_opt <- c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout")
hidden_opt <- list(c(100,100),c(200,150),c(500,500,500))
l1_opt <- c(0,1e-3,1e-5)
l2_opt <- c(0,1e-3,1e-5)

hyper_params <- list( activation=activation_opt,
                      hidden=hidden_opt,
                      l1=l1_opt,
                      l2=l2_opt )

hyper_params <- list(
  activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout"), 
  hidden = list(c(5, 5, 5, 5, 5), c(10, 10, 10, 10), c(50, 50, 50), c(100, 100, 100)),
  epochs = c(50, 100, 200),
  l1 = c(0, 0.00001, 0.0001), 
  l2 = c(0, 0.00001, 0.0001),
  rate = c(0, 01, 0.005, 0.001),
  rate_annealing = c(1e-8, 1e-7, 1e-6),
  rho = c(0.9, 0.95, 0.99, 0.999),
  epsilon = c(1e-10, 1e-8, 1e-6, 1e-4),
  momentum_start = c(0, 0.5),
  momentum_stable = c(0.99, 0.5, 0),
  input_dropout_ratio = c(0, 0.1, 0.2),
  max_w2 = c(10, 100, 1000, 3.4028235e+38)
)


#set search criteria
search_criteria <- list(strategy = "RandomDiscrete", max_models=20)

search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = 100,
                        max_runtime_secs = 900,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 15,
                        seed = 42)

#train model
s <- proc.time()
dl_grid <- h2o.grid("deeplearning"
                    ,grid_id = "deep_learn"
                    ,hyper_params = hyper_params
                    ,search_criteria = search_criteria
                    ,training_frame = trainh2o
                    ,x=x
                    ,y=y
                    ,nfolds = 10
                    ,epochs = 1000)
e <- proc.time()
d <- e - s
d
dl_grid <- h2o.grid(algorithm = "deeplearning", 
                    x = x,
                    y = y,
                    #weights_column = weights,
                    grid_id = "dl_grid",
                    training_frame = trainh2o,
                    validation_frame = testh2o,
                    nfolds = 25,                           
                    fold_assignment = "Stratified",
                    hyper_params = hyper_params,
                    search_criteria = search_criteria,
                    seed = 42
)
e <- proc.time()
d <- e - s
d

#get best model
d_grid <- h2o.getGrid("deep_learn",sort_by = "accuracy")
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])
h2o.performance (best_dl_model,xval = T) #CV Accuracy - 84.7%

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,xval = T) #84.5 % CV accuracy
h2o.varimp(my_gbm)
gc()

h2o.no_progress()
h2o.shutdown()
h2o.clusterStatus()
h2o.clusterInfo()

###############               ###############
# ============       GBM       ============ #
###############               ###############
# https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.Rmd
#https://www.kaggle.com/mmudaliar/feature-selection-in-h2o-gbm

## Start cluster with all available threads
h2o.init(nthreads = -1)
#localH2o <- h2o.init(nthreads = -1, max_mem_size = "20G")


## the response variable is an integer, we will turn it into a categorical/factor for binary classification
trainh2o[["pesocat"]] <- as.factor(trainh2o[["pesocat"]]) 

testh2o[["pesocat"]] <- as.factor(testh2o[["pesocat"]]) 

#set variables
y <- "pesocat"
x <- setdiff(colnames(trainh2o),y)

hyper_params = list( 
  ## restrict the search to the range of max_depth established above
  #max_depth = seq(minDepth = 9, maxDepth = 27,1),
  max_depth = seq(9, 27,1),
  
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),                                             
  
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),                                         
  
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.2,1,0.01),                                
  
  ## search a large space of how column sampling per split should change as a function of the depth of the split
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
  
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0,log2(nrow(trainh2o))-1,1), 
  
  ## search a large space of the number of bins for split-finding for continuous and integer columns
  nbins = 2^seq(4,10,1),                                                     
  
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4,12,1),                                                
  
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
  
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)


search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  
  ## limit the runtime to 60 minutes
  max_runtime_secs = 3600,         
  
  ## build no more than 100 models
  max_models = 100,                  
  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1234,                        
  
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)

s <- proc.time()
grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  
  ## which algorithm to run
  algorithm = "gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "final_grid", 
  
  ## standard model parameters
  x = x, 
  y = y, 
  training_frame = trainh2o, 
  validation_frame = testh2o,
  
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 3600,                                                 
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)
e <- proc.time()
d <- e - s
d

## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)

#get best model
d_grid <- h2o.getGrid("final_grid",sort_by = "accuracy")
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])
h2o.performance (best_dl_model,xval = T) #CV Accuracy - 84.7%

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,xval = T) #84.5 % CV accuracy
h2o.varimp(my_gbm)
gc()


###############                         ###############
# ============       Random Forest       ============ #
###############                         ###############
#https://uc-r.github.io/random_forests

## Start cluster with all available threads
h2o.init(nthreads = -1)
#localH2o <- h2o.init(nthreads = -1, max_mem_size = "20G")

# hyperparameter grid
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(15, 35, by = 10),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(1, 5, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .75)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "AUC",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 30*60
)

# build grid search 
s <- proc.time()
set.seed(2512)
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = x, 
  y = y, 
  training_frame = trainh2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = search_criteria
)
e <- proc.time()
d <- e - s
d

# collect the results and sort by our model performance metric of choice
grid_perf2 <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "auc", 
  decreasing = FALSE
)
print(grid_perf2)

# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf2@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let's evaluate the model performance on a test set
ames_test.h2o <- as.h2o(ames_test)
best_model_perf <- h2o.performance(model = best_model, newdata = ames_test.h2o)

# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()

