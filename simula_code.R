rm(list = ls())
library(data.table)
library(tidyverse)
library(parallel)
library(doParallel)


database <- data.table::fread("E:\\base_SISVAN_CAD.csv", na.strings = 'NA', drop = 'V1') %>% data.frame()

database <- database %>% select(-gravidez, -dens_domic, -score, -nasc_antes, -peso_ebpn)
database <- database %>% select(pesocat,everything()) %>% data.frame()


completedData <- database
rm(database)
sample_0 <- completedData[completedData$pesocat == 0, c(1:8,14:17,19:20,165)]
sample_1 <- completedData[completedData$pesocat == 1, c(1:8,14:17,19:20,165)]

sample_0$parto<- recode(sample_0$parto, "1" = 0, "2" = 1)
sample_1$parto<- recode(sample_1$parto, "1" = 0, "2" = 1)

sample_0 <- subset(sample_0, qtdgestant != 99 & qtdpartnor != 99 & qtdpartces != 99 &
                   consprenat != 99)

sample_1 <- subset(sample_1, qtdgestant != 99 & qtdpartnor != 99 & qtdpartces != 99 &
                     consprenat != 99)

var_factor <- c("locnasc","estcivmae", "escmae", "racacormae", "consultas"
                )
var_normalize <- c("idademae", "qtdgestant", "qtdpartnor", "qtdpartces",
                   "semagestac", "consprenat")

for(i in var_factor){
  sample_0[[i]] <- as.factor(sample_0[[i]])
  sample_1[[i]] <- as.factor(sample_1[[i]])
}

normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

for(j in var_normalize){
  sample_0[[j]] <- normalize(sample_0[[j]])
}




dat1 <- data.table::fread("E:\\Simulation\\base_simula2954.csv", na.strings = 'NA') %>% data.frame()
dat2 <- data.table::fread("E:\\Simulation\\base_simula5486.csv", na.strings = 'NA') %>% data.frame()
dat3 <- data.table::fread("E:\\Simulation\\base_simula11816.csv", na.strings = 'NA') %>% data.frame()
dat4 <- data.table::fread("E:\\Simulation\\base_simula26586.csv", na.strings = 'NA') %>% data.frame()
dat5 <- data.table::fread("E:\\Simulation\\base_simula39245.csv", na.strings = 'NA') %>% data.frame()

nomes <- c("var", "interation", "beta", "se", "pvalue", "CI2.5", "CI97.5",
           "Odds_CI2.5", "Odds_CI97.5","unbalanced")

dat1$unbalanced <- 'unb50'; dat2$unbalanced <- 'unb35'; dat3$unbalanced <- 'unb20';
dat4$unbalanced <- 'unb10'; dat5$unbalanced <- 'unb07'

colnames(dat1) <- nomes; colnames(dat2) <- nomes; colnames(dat3) <- nomes
colnames(dat4) <- nomes; colnames(dat5) <- nomes


base_analise <- data.frame(dat1)
base_analise <- rbind.data.frame(dat1, dat2, dat3, dat4, dat5)

rm(dat1); rm(dat2); rm(dat3); rm(dat4); rm(dat5)

var_plot <- as.factor(unique(base_analise$var))

for(j in var_plot){
  
  nomes_plot <- c("beta", "se", "pvalue", "CI2.5", "CI97.5",
                  "Odds_CI2.5", "Odds_CI97.5")
  print(j)
  
  filtro <- filter(base_analise, var == j)
  
  for(k in nomes_plot){
    print(k)
    var_k <- noquote(k)
    
    ggplot(data = filtro) + 
      geom_boxplot (mapping = aes(x = unbalanced, y = filtro[,var_k])) +
      ggtitle(j) + ylab(var_k)
    ggsave(paste("E:\\Simulation\\graphcs2",j,"_",k,".jpeg"))
    
  }
  
}

base_analise$simula <- rep(seq(1:1000), each = 364)

a <- base_analise %>%
  group_by(simula) %>%
  filter(var == "Intercept") %>% 
  dplyr::summarize(Mean = mean(beta, na.rm=TRUE))


var_plot <- as.factor(unique(base_analise$var))

for(i in var_plot){
  
  filtro2 <- base_analise %>% group_by(simula) %>% filter(var == i) %>% 
    mutate(media = mean(beta, na.rm = T),
           se_media = mean(se, na.rm = T),
           ic_max = media + 1.96*se_media,
           ic_min = media - 1.96*se_media) %>% distinct(simula, media, se_media, ic_max, ic_min)
  
  #require(ggplot2)
  ggplot(filtro2, aes(x = simula, y = media)) +
    geom_point(size = 4) +
    geom_errorbar(aes(ymax = ic_max, ymin = ic_min))+
    ylab(i) + theme(text = element_text(size=40))
  ggsave(paste("E:\\Simulation\\graphcs4\\",i, ".jpeg"), width = 20, height = 20)
}

