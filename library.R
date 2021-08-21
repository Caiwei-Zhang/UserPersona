packages <- c("text2vec", "caret", "data.table", "tidyverse", "ggplot2", "ggpubr", "lightgbm")
lapply(packages, function(pkg) {if(!require(pkg)) install.packages(pkg)})

library(data.table)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(text2vec)
library(caret)
library(lightgbm)
library(ROSE)


