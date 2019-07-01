rm(list=ls())
library(data.table)
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(ggplot2)
library(DALEX)

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

df2 <- data.table::fread("/Dados/BD_ORIGINAL/base_completa.csv",sep = "auto", header= TRUE, 
                         na.strings = "NA") %>% data.frame()
df3 <- df2[,var_select]
rm(df2); rm(var_select)

#== Exclude continuos variable with near zero variance ==#
#df <- df3[,-nearZeroVar(df3)]

#== Create Partition ==#
ind <- createDataPartition(df3$pesocat, p = .85, list = FALSE)
training <- df3[ind,]
testing <- df3[-ind,]
rm(df3)

set.seed(123456789) 

#== We can see the relationship between the tuning parameters and the area under the ROC curve
fit_control <- trainControl(method = 'repeatedcv', # repeate the fold number of according next paramenters 
                            number = 10, # Number of Fold
                            #repeats = 1, # Number of Fold repetitions 
                            search = 'random', # Describing how the tuning parameter
                            classProbs = TRUE, # Just used TRUE to classification a logical; should class probabilities be computed for 
                                               # classification models (along with predicted values) in each resample.
                            summaryFunction = twoClassSummary) # A function to compute performance metrics across resamples
                                                               # twoClassSummary computes sensitivity, specificity and the area under the ROC curve.
                            
 
#============= DataSet =============#
x_training <- training[,-1]  
y_training <- as.factor(recode(training$pesocat, '1' = 'B', '0' = 'N')) # Converts to factor
rm(training); rm(ind)

#=============                                         =============#
#============= RDA (Regularized Discriminant Analysis) =============#
#=============                                         =============#

#== Finding the number of cores in sytem
no_cores <- detectCores(logical = FALSE)
no_threads <- detectCores(logical = TRUE)
cat("CPU with", no_cores,"cores and", no_threads, "threads detected.\n")

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  rda_fit <- train(x_training, y_training,
                   method = 'rda',
                   importance = TRUE,
                   trControl = fit_control,
                   verbose = FALSE,
                   lunelength = 9,
                   metric = "ROC")
)

stopCluster(cl)
registerDoSEQ()
gc()

rda_result1 <- data.frame(rda_fit$results)
rda_result2 <- data.frame(rda_fit$bestTune)
rda_pred <- predict(rda_fit, newdata = testing, type = "prob")
rda_imp <- varImp(rda_fit)$importance
rda_imp <- data.frame(row.names(rda_imp), rda_imp)

data.table::fwrite(rda_result1, file = "/home/rda_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(rda_result2, file = "/home/rda_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(rda_pred, file = "/home/rda_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(rda_imp, file = "/home/rda_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)
#=============                                   =============#
#============= nnet (Neural Network Feedforward) =============#
#=============                                   =============#
nnet_fitcontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 1, classProbs = TRUE,
                                verboseIter = TRUE, summaryFunction = twoClassSummary,
                                preProcOptions = list(thresh = 0.75, ICAcomp =3, k =5),
                                allowParallel = TRUE, returnData = FALSE)

#== Finding the number of cores in sytem
no_cores <- detectCores(logical = FALSE)
no_threads <- detectCores(logical = TRUE)
cat("CPU with", no_cores,"cores and", no_threads, "threads detected.\n")

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  nnet_fit <- train(x_training, y_training,
                    method = 'nnet',
                    #importance = TRUE,
                    trControl = nnet_fitcontrol,
                    tuneGrid = expand.grid(size = c(10), decay = c(0.001)),
                    metric = "ROC")
)

stopCluster(cl)
registerDoSEQ()
gc()

nnet_result1 <- data.frame(nnet_fit$results)
nnet_result2 <- data.frame(nnet_fit$bestTune)
nnet_pred <- predict(nnet_fit, newdata = testing, type = "prob")
nnet_imp <- varImp(nnet_fit)$importance
nnet_imp <- data.frame(row.names(nnet_fit), nnet_fit)

data.table::fwrite(nnet_result1, file = "/home/nnet_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(nnet_result2, file = "/home/nnet_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(nnet_pred, file = "/home/nnet_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(nnet_imp, file = "/home/nnet_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

#=============                                   =============#
#============= XGBoost (Extreme Gradient Boost)  =============#
#=============                                   =============#

# Model Parameters Grid
parameters_grid <- expand.grid(nrounds = 100, max_depth = 3, eta = 0.01, gamma =1, colsample_bytree = 0.5,
                               min_child_weight = 2, subsample =1)

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  xgboost_fit <- train(x_training, y_training,
                       method = 'xgbTree', metric = "ROC",
                       #importance = TRUE,
                       trControl = fit_control,
                       tuneGrid = parameters_grid
                       )
)

stopCluster(cl)
registerDoSEQ()
gc()

xgboost_result1 <- data.frame(xgboost_fit$results)
xgboost_result2 <- data.frame(xgboost_fit$bestTune)
xgboost_pred <- predict(xgboost_fit, newdata = testing, type = "prob")
xgboost_imp <- varImp(xgboost_fit)$importance
xgboost_imp <- data.frame(row.names(xgboost_fit), xgboost_fit)

data.table::fwrite(xgboost_result1, file = "/home/xgboost_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(xgboost_result2, file = "/home/xgboost_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(xgboost_pred, file = "/home/xgboost_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(xgboost_imp, file = "/home/xgboost_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

rm(xgboost_fit); rm(xgboost_result1); rm(xgboost_result2); rm(xgboost_pred); rm(xgboost_imp)

#=============             =============#
#============= Naive Bayes =============#
#=============             =============#

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  nb_fit <- train(x_training, y_training,
                        method = 'nb', metric = "ROC",
                        verbose = FALSE,
                        #importance = TRUE,
                        trControl = fit_control
                        
  )
)

stopCluster(cl)
registerDoSEQ()
gc()

nb_result1 <- data.frame(nb_fit$results)
nb_result2 <- data.frame(nb_fit$bestTune)
nb_pred <- predict(nb_fit, newdata = testing, type = "prob")
nb_imp <- varImp(nb_fit)$importance
nb_imp <- data.frame(row.names(nb_fit), nb_fit)

data.table::fwrite(nb_result1, file = "/home/nb_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(nb_result2, file = "/home/nb_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(nb_pred, file = "/home/nb_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(nb_imp, file = "/home/nb_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

rm(nb_fit); rm(nb_result1); rm(nb_result2); rm(nb_pred); rm(nb_imp)

#=============             =============#
#============= ElasticNet  =============#
#=============             =============#

# Model Parameters Grid
parameters_grid <- expand.grid(.alpha = seq(0, 1, by = 0.5), .lambda = seq(0, 0.2, by = 0.1))

#fit_control_elnet <- trainControl(method = "cv", number = 5, classProbs = TRUE, verbose = FALSE, returnData = FALSE, search = "random")

#== Setup cluster
#cl <- makeCluster(no_threads)
cl <- makeCluster(40)
registerDoParallel(cl)

system.time(
  elnet_fit <- train(x = data.matrix(x_training), y = y_training,
                  method = 'glmnet', metric = "ROC",
                  family = "binomial",
                  #importance = TRUE,
                  trControl = fit_control, tuneGrid = parameters_grid,
                  standardize = FALSE,
                  maxit = 1000
                  
  )
)
#eNetModel <- train(Class ~ ., data=trainData, method = "glmnet", metric="ROC", trControl = fitControl, family="binomial", tuneLength=5)
stopCluster(cl)
registerDoSEQ()
gc()

elnet_fit_result1 <- data.frame(elnet_fit$results)
elnet_fit_result2 <- data.frame(elnet_fit$bestTune)
elnet_fit_pred <- predict(elnet_fit, newdata = testing, type = "prob")
elnet_fit_imp <- varImp(elnet_fit)$importance
elnet_fit_imp <- data.frame(row.names(elnet_fit_imp), elnet_fit_imp)

data.table::fwrite(elnet_fit_result1, file = "/home/elnet_fit_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(elnet_fit_result2, file = "/home/elnet_fit_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(elnet_fit_pred, file = "/home/elnet_fit_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(elnet_fit_imp, file = "/home/elnet_fit_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

rm(elnet_fit); rm(elnet_fit_result1); rm(elnet_fit_result2); rm(elnet_fit_pred); rm(elnet_fit_imp)

#=============                     =============#
#============= Logistic Regression =============#
#=============                     =============#

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  lr_fit <- train(x_training, y_training,
                  method = 'glmnet', metric = "ROC",
                  #verbose = FALSE,
                  family = 'binomial',
                  #importance = TRUE,
                  trControl = fit_control,
                  tuneGrid=expand.grid(parameter=c(0.001, 0.01, 0.1, 1,10,100, 1000))
                  
  )
)

stopCluster(cl)
registerDoSEQ()
gc()

lr_fit_result1 <- data.frame(lr_fit$results)
lr_fit_result2 <- data.frame(lr_fit$bestTune)
lr_fit_pred <- predict(lr_fit, newdata = testing, type = "prob")
lr_fit_imp <- varImp(lr_fit)$importance
lr_fit_imp <- data.frame(row.names(lr_fit_imp), lr_fit_imp)

data.table::fwrite(lr_fit_result1, file = "/home/lr_fit_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(lr_fit_result2, file = "/home/lr_fit_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(lr_fit_pred, file = "/home/lr_fit_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(lr_fit_imp, file = "/home/lr_fit_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

rm(lr_fit); rm(lr_fit_result1); rm(lr_fit_result2); rm(lr_fit_pred); rm(lr_fit_imp)










#=============                                     =============#
#============= LVQ - Learning Vector Quantization  =============#
#=============                                     =============#

fit_control_lvq <- trainControl(method = "cv", number = 10, returnResamp = "none", search = "random")
control <- trainControl(method="repeatedcv", number=10, repeats=2, returnResamp = "none")
# twoClassSummary computes sensitivity, specificity and the area under the ROC curve.

#x_training <- as.data.frame(x_training)

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  lvq_fit <- train(x_training, y_training,
                     method = 'lvq', metric = "Accuracy",
                     verbose = FALSE,
                     #importance = TRUE,
                     trControl = control
                     
  )
)

stopCluster(cl)
registerDoSEQ()
gc()

lvq_fit_result1 <- data.frame(lvq_fit$results)
lvq_fit_result2 <- data.frame(lvq_fit$bestTune)
lvq_fit_pred <- predict(lvq_fit, newdata = testing, type = "prob")
lvq_fit_imp <- varImp(lvq_fit)$importance
lvq_fit_imp <- data.frame(row.names(lvq_fit), lvq_fit)

data.table::fwrite(lvq_fit_result1, file = "/home/lvq_fit_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(lvq_fit_result2, file = "/home/lvq_fit_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(lvq_fit_pred, file = "/home/lvq_fit_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(lvq_fit_imp, file = "/home/lvq_fit_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

rm(lvq_fit); rm(lvq_fit_result1); rm(lvq_fit_result2); rm(lvq_fit_pred); rm(lvq_fit_imp)

#=============               =============#
#============= Random Forest =============#
#=============               =============#

fit_control_rf <- trainControl(method = 'repeatedcv', # repeate the fold number of according next paramenters 
                            number = 10, # Number of Fold
                            #repeats = 1, # Number of Fold repetitions 
                            search = 'random', # Describing how the tuning parameter
                            classProbs = TRUE, # Just used TRUE to classification a logical; should class probabilities be computed for 
                            # classification models (along with predicted values) in each resample.
                            returnData = FALSE,
                            
                            summaryFunction = twoClassSummary) # A function to compute performance metrics across resamples
# twoClassSummary computes sensitivity, specificity and the area under the ROC curve.

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  rf_fit <- train(x_training, y_training,
                   method = 'rf', metric = "ROC",
                   verbose = FALSE,
                   #importance = TRUE,
                   trControl = fit_control_rf
                   
  )
)

stopCluster(cl)
registerDoSEQ()
gc()

rf_fit_result1 <- data.frame(rf_fit$results)
rf_fit_result2 <- data.frame(rf_fit$bestTune)
rf_fit_pred <- predict(rf_fit, newdata = testing, type = "prob")
rf_fit_imp <- varImp(rf_fit)$importance
rf_fit_imp <- data.frame(row.names(rf_fit), rf_fit)

data.table::fwrite(rf_fit_result1, file = "/home/rf_fit_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(rf_fit_result2, file = "/home/rf_fit_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(rf_fit_pred, file = "/home/rf_fit_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(rf_fit_imp, file = "/home/rf_fit_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = TRUE, col.names = TRUE)

rm(rf_fit); rm(rf_fit_result1); rm(rf_fit_result2); rm(rf_fit_pred); rm(rf_fit_imp)




#============= GBM (Stochastic Gradient Boosting) =============#
# == Parameter Tunning 
gbm_grid <- expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:20)*10,
                        shrinkage = 0.01,
                        n.minobsinnode = 20)

#== Finding the number of cores in sytem
no_cores <- detectCores(logical = FALSE)
no_threads <- detectCores(logical = TRUE)
cat("CPU with", no_cores,"cores and", no_threads, "threads detected.\n")

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  gbm_fit <- train(x_training, y_training,
                   method = 'gbm',
                   trControl = fit_control,
                   verbose = FALSE,
                   tuneGrid = gbm_grid,
                   metric = "ROC")
)

stopCluster(cl)
registerDoSEQ()
gc()

gbm_result1 <- data.frame(gbm_fit$results)
gbm_result2 <- data.frame(gbm_fit$bestTune)
gbm_pred <- predict(gbm_fit, newdata = testing, type = "prob")
gbm_imp <- data.frame(summary(gbm_fit))

data.table::fwrite(gbm_result1, file = "/home/gbm_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)
  
data.table::fwrite(gbm_result2, file = "/home/gbm_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(gbm_pred, file = "/home/gbm_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(gbm_imp, file = "/home/gbm_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

#============= SVM (Suport Vector Machine) =============#  
# == Parameter Tunning 
svm_grid <- expand.grid(.sigma = c(0.0577), .C = c(2.21049))

#== Finding the number of cores in sytem
no_cores <- detectCores(logical = FALSE)
no_threads <- detectCores(logical = TRUE)
cat("CPU with", no_cores,"cores and", no_threads, "threads detected.\n")

TrainCtrl1 <- trainControl(method = "repeatedcv", number = 5, classProbs = TRUE, verbose = FALSE, returnData = FALSE)

#== Setup cluster
cl <- makeCluster(no_threads)
registerDoParallel(cl)

system.time(
  svm_fit <- train(x_training, y_training,
                   method = 'svmRadial',
                   importance = TRUE,
                   trControl = TrainCtrl1,
                   verbose = FALSE,
                   tuneGrid = svm_grid,
                   metric = "Accuracy")
)




modelSvmRRB <- train(X, Y, method="svmRadial", trControl=TrainCtrl1,tuneGrid = SVMgrid,preProc = c("scale","YeoJohnson"), verbose=FALSE)


stopCluster(cl)
registerDoSEQ()
gc()

svm_result1 <- data.frame(svm_fit$results)
svm_result2 <- data.frame(svm_fit$bestTune)
svm_pred <- predict(svm_fit, newdata = testing, type = "prob")
svm_imp <- varImp(svm_fit)$importance
svm_imp <- data.frame(row.names(svm_imp), svm_imp)

data.table::fwrite(svm_result1, file = "/home/svm_result1.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(svm_result2, file = "/home/svm_result2.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)  

data.table::fwrite(svm_pred, file = "/home/svm_pred.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

data.table::fwrite(svm_imp, file = "/home/svm_imp.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)




#===========================
p_fun <- function(object, newdata){predict(object, newdata=newdata, type="prob")[,2]}
y_training2 <- as.numeric(recode(y_training, 'B' = '1', 'N' = '0')) # Converts to factor
yTest <- y_training2
rm(y_training2)

explainer_classif_nnet <- DALEX::explain(nnet_fit, label = "nnet",
                                         data = x_training, y = yTest,
                                         predict_function = p_fun)
#== Model performance
mp_classif_nnet <- model_performance(explainer_classif_nnet)
plot(mp_classif_nnet)
plot(mp_classif_nnet, geom = "boxplot")

#== Variable importance
vi_classif_nnet<- variable_importance(explainer_classif_nnet, loss_function = loss_root_mean_square)
plot(vi_classif_nnet)
data.table::fwrite(vi_classif_nnet, file = "/home/vi_classif_nnet.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

#== Partial Depedence Plot
pdp_classif_nnet  <- variable_response(explainer_classif_nnet, variable = "escmae", type = "pdp")
plot(pdp_classif_nnet)
data.table::fwrite(pdp_classif_nnet, file = "/home/pdp_classif_nnet.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

#== Acumulated Local Effects plot
ale_classif_nnet <- variable_response(explainer_classif_nnet, variable = "escmae", type = "ale")
plot(ale_classif_nnet)
data.table::fwrite(ale_classif_nnet, file = "/home/ale_classif_nnet.csv", quote = "auto", sep = ";", dec = ",",
                   row.names = FALSE, col.names = TRUE)

mpp_regr_nnet <- variable_response(explainer_classif_nnet, variable = "escmae", type = "factor")
plot(mpp_regr_nnet)

