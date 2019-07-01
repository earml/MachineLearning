rm(list = ls())
library(data.table)
library(tidyverse)
library(Boruta)
#library(glmnet)
#library(lars)

database <- data.table::fread("E:\\base_SISVAN_CAD.csv", na.strings = 'NA', drop = 'V1') %>% data.frame()
#convert <- c(1, 3:8, 13, 14, 75)
#database[,convert] <- data.frame(apply(database[convert], 2, as.factor))
database <- database %>% select(-gravidez, -dens_domic, -score, -nasc_antes, -peso_ebpn)
database <- database %>% select(pesocat,everything()) %>% data.frame()


#############
#===================== BORUTA =====================#
#############
set.seed(123456789)
system(
boruta_model <- Boruta(pesocat~., data = database[10000:15000,], doTrace = 2, maxRuns = 20)
)
gc()
importance_classification <- data.frame(boruta_model$finalDecision) # Variable classification (Rejected, Tentative, Confirmed)
importance_history <- data.frame(boruta_model$ImpHistory) # Importance history
stats <- attStats(boruta_model) # importance statistics of model (mean,median, min, max, normahits, decision)
stats$name <- row.names(stats)

data.table::fwrite(importance_classification, file = "E:\\featureSelection\\Resultados\\varselecinadas_boruta.csv",quote = "auto",sep = ";",dec = ".", row.names = T, col.names = T)
data.table::fwrite(importance_history, file = "E:\\featureSelection\\Resultados\\importance_history_boruta.csv",quote = "auto",sep = ";",dec = ".", row.names = F, col.names = T)
data.table::fwrite(stats, file = "E:\\featureSelection\\Resultados\\stats_boruta.csv",quote = "auto",sep = ";",dec = ".", row.names = T, col.names = T)

plotar_stats <- stats %>% filter(decision %in% c('Confirmed')) 
plotar_stats$name <- factor(plotar_stats$name, levels = plotar_stats$name[order(plotar_stats$meanImp)])

ggplot(data = plotar_stats) +
  geom_col(mapping = aes(x = name, y = meanImp), position = "stack") + coord_flip() +
  ggtitle("Random Forest")
ggsave("randomForest.pdf")
ggsave("randomForest.png")
ggsave("randomForest.jpeg")

#decision_final <- TentativeRoughFix(boruta_model)
#ggplot(data = stats) +
#  geom_point(mapping = aes(x = name, y = meanImp, color = stats$decision)) +
#  theme(axis.ticks.x =  element_blank())  +

