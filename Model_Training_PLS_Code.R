#windows
#setwd("C:\\Users\\John\\OneDrive\\Masters Thesis Analysis\\DataWindows\\Project")

#mac
setwd("/Users/johnlunalo/Library/CloudStorage/OneDrive-Personal/Masters Thesis Analysis/DataWindows/Project")



Data <- read.csv(file = "model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

Data$CLASS <- ifelse(Data$CLASS == "BRCA", 1, ifelse(Data$CLASS == "COAD", 2, ifelse(Data$CLASS == "LUAD", 3, ifelse(Data$CLASS == "OV", 4, 5))))
table(Data$CLASS)

library(tidyverse)
library(caret)
library(pls)
library(ggplot2)
library(vip)
library(chillR)
library(mdatools)
library(glmnet)



# Set the target variable
target <- "CLASS"


set.seed(123) 
trainIndex <- createDataPartition(Data$CLASS, p = 0.70, list = FALSE)
trainData <- Data[trainIndex, ]
testData <- Data[-trainIndex, ]


library(doParallel)
nocores <- detectCores() - 2
cl <- makeCluster(nocores)
registerDoParallel(cl)

# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	    


fitControl <- trainControl(method="CV",   # 10 folds cross validation
                        number = 10,
                        savePredictions = TRUE,
                        allowParallel = TRUE,
                        p=0.85,
                        preProc = c("scale")
)


pls_fit1<-caret::train(CLASS ~ ., data = Data, method ="pls", trControl=fitControl
                      )
ggplot(pls_fit1, aesthetics =aes(fill ="#0083b8")) + theme_minimal() + xlab("Principal Componets") + ggtitle("Scree Plot of RMSE Against Principal Componets ")+
  theme(plot.title = element_text(hjust = 0.5, colour = "black", size = 22), axis.title = element_text(colour = "black", size = 16))


# varimp_vec_loop <- seq(2.9,4, 0.05)
# accuracy = c()
# # 
# for(i in varimp_vec_loop){
#   varimp_vec <- varImp(pls_fit1, ncomp = 3)
#   #sum(varimp_vec$importance>i)
#   #summary(pls_fit1)
# 
# 
#   comp_loadings <- pls_fit1$finalModel$loadings[1:14899, 1:2]
# 
#   #comp_loadings1 <- data.frame(cols = unlist(dimnames(comp_loadings)[1]))
# 
#   #comp_loadings1$trial <- as.vector(apply(comp_loadings, MARGIN = 1, FUN = function(x){max(abs(x))}))
# 
#   # vip::vip(pls_fit1, num_features=20, aesthetics =aes(fill ="#0083b8"))+theme_minimal()+
#   #   ylab("Importance") + ggtitle("Variable Importance for Top 20 Features ")+
#   #   theme(plot.title = element_text(hjust = 0.5, colour = "black", size = 22),
#   #         axis.title = element_text(colour = "black", size = 16))+coord_flip()
#   # ggsave("vis.png")
# 
#   vip_sele <- vip::vip(pls_fit1)# ten most important variables
# 
# 
#   # select variables with VIP score above 1
# 
#   varimp_vec_df <- varimp_vec[[1]]
#   varimp_vec_df$var_name <- rownames(varimp_vec_df)
#   selected_vars <- varimp_vec_df[varimp_vec_df$Overall>i, "var_name"]
#   selected_vars_pls<- selected_vars
# 
#   #unique(c(selected_vars_pls[selected_vars_pls %in% row.names(LassoCof_NonZero)], row.names(LassoCof_NonZero)[ row.names(LassoCof_NonZero) %in% selected_vars_pls]))
#   #selected_vars1 <- rbind(selected_vars, colSums(selected_vars)>0)[3, ]
# 
# 
#   ###Find another list features correlated with the first 2 PCs
# 
#   selected_vars1_df <- data.frame(features = selected_vars)
# 
#   write.csv(x = selected_vars1_df, file = "plsselected.csv", row.names = TRUE)
#   #########################
# 
# 
#   ############Save data for the selected genes
#   plsselected <- read.csv(file = "plsselected.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)
#   Data <- read.csv(file = "model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)
# 
# 
#   plsData <- Data[, names(Data)%in%plsselected$features]
#   plsData$CLASS <- Data$CLASS[match(row.names(plsData), row.names(Data))]
# 
#   ############Save Final Data from Lasso
#   plsData$CancerType <- plsData$CLASS
#   plsData <- plsData[, -which(names(plsData) =="CLASS")]
#   Data <- plsData
# 
# 
# 
# 
# 
#   trainIndex <- createDataPartition(Data$CancerType,p=.70,list=FALSE)
#   trainData <- Data[trainIndex,]
#   testData  <- Data[-trainIndex,]
#   Xtrain <- as.matrix(trainData[,-which(names(trainData) %in% c("CancerType"))])
#   Ytrain <- as.matrix(trainData[,which(names(trainData) %in% c("CancerType"))])
#   Ytrain[,1]<- as.factor(Ytrain[,1])
# 
#   Xtest <- as.matrix(testData[,-which(names(testData) %in% c("CancerType"))])
#   Ytest <- as.matrix(testData[,which(names(testData) %in% c("CancerType"))])
#   Ytest[,1]<- as.factor(Ytest[,1])
# 
# 
#   set.seed(849)
# 
# 
#   library(doParallel)
#   nocores <- detectCores() - 1
#   cl <- makeCluster(nocores)
#   registerDoParallel(cl)
# 
#   # Training SVM Models
#   library(dplyr)         # Used by caret
#   library(kernlab)       # support vector machine
#   library(pROC)	       # plot the ROC curves
# 
#   # Define train control
#   ctrlXGB <- trainControl(
#     method = "cv",
#     number = 5,
#     savePredictions = TRUE,
#     classProbs = TRUE,
#     allowParallel = TRUE
#   )
#   
#   # Perform grid search
#   XGBModel.tune <- train(
#     x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
#     y = trainData$CancerType,
#     method = "xgbTree",   # XGBoost
#     metric = "Accuracy",
#     trControl = ctrlXGB
#   )
#   
#   # 
#   # Tuning parameter 'gamma' was held constant at a value of 0
#   # Tuning parameter 'colsample_bytree' was held constant at a value of 1
#   # Tuning parameter 'min_child_weight' was
#   # held constant at a value of 1
#   # Tuning parameter 'subsample' was held constant at a value of 1
#   # Accuracy was used to select the optimal model using the largest value.
#   # The final values used for the model were nrounds = 200, max_depth = 2, eta = 0.3, gamma = 0, colsample_bytree = 1, min_child_weight = 1 and subsample = 1.
#   
#   boostmodel.pred <- predict(XGBModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
#   boost.tab = table(pred = boostmodel.pred, true = testData[,c("CancerType")])
#   boostmodel.Conf = confusionMatrix(as.factor(boostmodel.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
#   boostmodel.Conf
#   accuracy <- c(accuracy, boostmodel.Conf$overall[1])
# }
# # 
#accuracy
# 
#df <- data.frame(accuracy = accuracy, varimp_vec_loop = varimp_vec_loop)
#write.csv(df, "vip_accuracy.csv")


df <- read.csv("vip_accuracy.csv")
############End of selecting Variables
selected_vip <- df[df$accuracy==max(df$accuracy), "varimp_vec_loop"][1]
selected_vip1 <- df[df$accuracy==max(df$accuracy), "varimp_vec_loop"][2]
selected_vip2 <- df[df$accuracy==max(df$accuracy), "varimp_vec_loop"][3]
ggplot(df, mapping =aes(x = varimp_vec_loop, y = accuracy)) + geom_line(color ="#0083b8")+
  
  theme_minimal() + xlab("Variable Importance") + ggtitle("Accuracy against VIP")+
  theme(plot.title = element_text(hjust = 0.5, colour = "black", size = 22), axis.title = element_text(colour = "black", size = 16))+
  geom_text(aes(x= selected_vip, y = max(accuracy)+.003, label = paste0(round(max(accuracy)*100, 2), "%")), size = 4)+
  #geom_text(aes(x= selected_vip1, y = max(accuracy)+.003, label = paste0(round(max(accuracy)*100, 2), "%")), size = 4)+
  geom_text(aes(x= selected_vip2, y = max(accuracy)+.003, label = paste0(round(max(accuracy)*100, 2), "%")), size = 4)

ggsave("vip_optimum.png")
#selected_vip <- df[df$accuracy==max(df$accuracy), "varimp_vec_loop"]

varimp_vec <- varImp(pls_fit1, ncomp = 3)
sum(varimp_vec$importance>selected_vip)
#summary(pls_fit1)


comp_loadings <- pls_fit1$finalModel$loadings[1:14899, 1:2]

#comp_loadings1 <- data.frame(cols = unlist(dimnames(comp_loadings)[1]))

#comp_loadings1$trial <- as.vector(apply(comp_loadings, MARGIN = 1, FUN = function(x){max(abs(x))}))

#vip::vip(pls_fit1, num_features=20, aesthetics =aes(fill ="#0083b8"))+theme_minimal()+
#   ylab("Importance") + ggtitle("Variable Importance for Top 20 Features ")+
#   theme(plot.title = element_text(hjust = 0.5, colour = "black", size = 22),
#         axis.title = element_text(colour = "black", size = 16))+coord_flip()
# ggsave("vis.png")

#vip_sele <- vip::vip(pls_fit1)# ten most important variables


# select variables with VIP score above 1

varimp_vec_df <- varimp_vec[[1]]
varimp_vec_df$var_name <- rownames(varimp_vec_df)
selected_vars <- varimp_vec_df[varimp_vec_df$Overall>selected_vip, "var_name"]
selected_vars_pls<- selected_vars

#unique(c(selected_vars_pls[selected_vars_pls %in% row.names(LassoCof_NonZero)], row.names(LassoCof_NonZero)[ row.names(LassoCof_NonZero) %in% selected_vars_pls]))
#selected_vars1 <- rbind(selected_vars, colSums(selected_vars)>0)[3, ]


###Find another list features correlated with the first 2 PCs

selected_vars1_df <- data.frame(features = selected_vars)

write.csv(x = selected_vars1_df, file = "plsselected.csv", row.names = TRUE)
#########################


############Save data for the selected genes
plsselected <- read.csv(file = "plsselected.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)
Data <- read.csv(file = "model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)


plsData <- Data[, names(Data)%in%plsselected$features]
plsData$CLASS <- Data$CLASS[match(row.names(plsData), row.names(Data))]

############Save Final Data from Lasso
plsData$CancerType <- plsData$CLASS
plsData <- plsData[, -which(names(plsData) =="CLASS")]

write.csv(x = plsData, file = "plsDataLatest.csv", row.names = TRUE)

##################################################################################################################

#partition the data
Data <- read.csv(file = "plsDataLatest.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)
#Data$CancerType <- ifelse(Data$CancerType == "BRCA", 1, ifelse(Data$CancerType == "COAD", 2, ifelse(Data$CancerType == "LUAD", 3, ifelse(Data$CancerType == "OV", 4, 5))))


trainIndex <- createDataPartition(Data$CancerType,p=.80,list=FALSE)
trainData <- Data[trainIndex,]
testData  <- Data[-trainIndex,]
Xtrain <- as.matrix(trainData[,-which(names(trainData) %in% c("CancerType"))])
Ytrain <- as.matrix(trainData[,which(names(trainData) %in% c("CancerType"))])
Ytrain[,1]<- as.factor(Ytrain[,1])

Xtest <- as.matrix(testData[,-which(names(testData) %in% c("CancerType"))])
Ytest <- as.matrix(testData[,which(names(testData) %in% c("CancerType"))])
Ytest[,1]<- as.factor(Ytest[,1])


set.seed(849)


#SVM

#trainData$CancerType <- as.factor(ifelse(trainData$CancerType == 1, "BRCA", ifelse(trainData$CancerType == 2, "COAD", ifelse(trainData$CancerType == 3, "LUAD", ifelse(trainData$CancerType == 4, "OV", "THCA")))))
#testData$CancerType <- as.factor(ifelse(testData$CancerType == 1, "BRCA", ifelse(testData$CancerType == 2, "COAD", ifelse(testData$CancerType == 3, "LUAD", ifelse(testData$CancerType == 4, "OV", "THCA")))))
library(doParallel)
nocores <- detectCores() - 1
cl <- makeCluster(nocores)
registerDoParallel(cl)

# Training SVM Models
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves

ctrlSVM <- caret::trainControl(method="CV",   # 10 folds cross validation
                               number = 10,
                               classProbs=TRUE,
                               savePredictions = TRUE,
                               allowParallel = TRUE,
                               sampling = "smote"
)


svmRadial.tune <- caret::train(x=as.data.frame(scale(trainData[-which(names(trainData) %in% c("CancerType"))])),
                               y= trainData$CancerType,
                               method = "svmRadial",   # Radial kernel
                               tuneLength = 5,					# 9 values of the cost function
                               #preProc = c("center","scale"),  # Center and scale data
                               metric="Accuracy",
                               trControl=ctrlSVM)


svmRadial.tune



svmRadial.Conf$overall[1]


svmRadial.pred <- predict(svmRadial.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
svmRadial.tab = table(pred = svmRadial.pred, true = testData[,c("CancerType")])
svmRadial.Conf = confusionMatrix(as.factor(svmRadial.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])

svmRadial.Conf

valuation_table_svmradial_pls <- t(svmRadial.Conf$byClass)
valuation_kapa_svmradial_pls <- svmRadial.tune$results

svmLinear.tune <- caret::train(x=scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                               y= trainData$CancerType,
                               method = "svmLinear",
                               tuneLength = 5,
                               #preProc = c("scale"),
                               metric="Accuracy",
                               trControl=ctrlSVM)	


svmLinear.tune

svmLinear.pred <- predict(svmLinear.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
svmLinear.tab = table(pred = svmLinear.pred, true = testData[,c("CancerType")])
svmLinear.Conf = confusionMatrix(svmLinear.pred, factor(testData[,c("CancerType")]), positive = levels(factor(testData[,c("CancerType")]))[1])
svmLinear.Conf

valuation_table_svmlinear_pls <- t(svmLinear.Conf$byClass)
valuation_kapa_svmlinear_pls <- svmLinear.tune$results
# 
svmPoly.tune <- caret::train(x=scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                             y= trainData$CancerType,
                             method = "svmPoly",
                             tuneLength = 5,
                             #preProc = c("scale"),
                             metric="Accuracy",
                             trControl=ctrlSVM)


svmPoly.tune

saveRDS(svmPoly.tune, "svmPoly_pls.RDS")
svmPoly.tune <- readRDS( "svmPoly_pls.RDS")

svmPoly.pred <- predict(svmPoly.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
svmPoly.tab = table(pred = svmPoly.pred, true = testData[,c("CancerType")])
svmPoly.Conf = confusionMatrix(as.factor(svmPoly.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
svmPoly.Conf

valuation_table_svmpoly_pls <- t(svmPoly.Conf$byClass)
valuation_kapa_svmploy_pls <- svmPoly.tune$results

#ANN
library(caret)
library(dplyr)         # Used by caret
library(nnet)       # support vector machine 
library(pROC)	       # plot the ROC curves

ctrlANN <- trainControl(method="CV",   # 10 folds cross validation
                        number = 10,
                        classProbs=TRUE,
                        allowParallel = TRUE,
                        savePredictions = TRUE,
                        sampling = "smote"
)

ANNModel.tune <- caret::train(x=scale(trainData[,-which(names(trainData) %in% c("CancerType"))]),
                              y= trainData$CancerType,
                              method = "nnet",   # Artificial nueral network
                              trace = FALSE, 
                              trControl=ctrlANN,
                              #preProcess = c('scale'),
                              metric="Accuracy")

ANNModel.tune

print(ANNModel.tune)
plot(ANNModel.tune)

ANNModel.pred <- predict(ANNModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
ANNModel.tab = table(pred = ANNModel.pred, true = testData[,c("CancerType")])
ANNModel.Conf = confusionMatrix(as.factor(ANNModel.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
ANNModel.Conf

valuation_table_ann_pls <- t(ANNModel.Conf$byClass)
valuation_kapa_ann_pls <- ANNModel.tune$results
#KNN
library(caret)
library(pROC)

ctrlKNN <- trainControl(method="CV",   # 10 folds cross validation
                        number = 10,
                        savePredictions = TRUE,
                        classProbs=TRUE,
                        allowParallel = FALSE,
                        sampling = "smote"
)

KNNModel.tune <- caret::train(x=scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                              y= trainData$CancerType,
                              method = "knn",   # K Nearest Neighbor
                              metric="Accuracy",
                              #preProcess = c('scale'),
                              trControl=ctrlKNN)

KNNModel.tune

print(KNNModel.tune)
plot(KNNModel.tune)

KNNModel.pred <- predict(KNNModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
KNNModel.tab = table(pred = KNNModel.pred, true = testData[,c("CancerType")])
KNNModel.Conf = confusionMatrix(as.factor(KNNModel.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
KNNModel.Conf


valuation_table_knn_pls <- t(KNNModel.Conf$byClass)
valuation_kapa_knn_pls <- KNNModel.tune$results

######Bagging using Random Forest

library(caret)
library(pROC)

ctrlRF <- trainControl(method = "CV",   # 10-fold cross-validation
                       number = 10,
                       savePredictions = TRUE,
                       classProbs = TRUE,
                       allowParallel = TRUE,
                       sampling = "smote"
)

RFModel.tune <- caret::train(x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                             y = trainData$CancerType,
                             method = "rf",   # Random Forest
                             metric = "Accuracy",
                             trControl = ctrlRF)

rfmodel.pred <- predict(RFModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
rf.tab = table(pred = rfmodel.pred, true = testData[,c("CancerType")])
rfmodel.Conf = confusionMatrix(as.factor(rfmodel.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
rfmodel.Conf

evaluation_table_rf_pls <- t(rfmodel.Conf$byClass)
valuation_kapa_rf_pls <- RFModel.tune$results

#Boosting using XgBoost

#install.packages("xgboost")

library(caret)
library(pROC)
library(xgboost)

# Define the tuning grid
nrounds <- 1000
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Define train control
ctrlXGB <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE,
  classProbs = TRUE,
  allowParallel = TRUE
)

# Perform grid search
XGBModel.tune <- train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
  y = trainData$CancerType,
  method = "xgbTree",   # XGBoost
  metric = "Accuracy",
  trControl = ctrlXGB,
  tuneGrid = tune_grid
)

# 
# Tuning parameter 'gamma' was held constant at a value of 0
# Tuning parameter 'colsample_bytree' was held constant at a value of 1
# Tuning parameter 'min_child_weight' was
# held constant at a value of 1
# Tuning parameter 'subsample' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were nrounds = 200, max_depth = 2, eta = 0.3, gamma = 0, colsample_bytree = 1, min_child_weight = 1 and subsample = 1.

boostmodel.pred <- predict(XGBModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
boost.tab = table(pred = boostmodel.pred, true = testData[,c("CancerType")])
boostmodel.Conf = confusionMatrix(as.factor(boostmodel.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
boostmodel.Conf

evaluation_table_boost_pls <- t(boostmodel.Conf$byClass)
valuation_kapa_boost_pls <- XGBModel.tune$results


#$$$$$$$$$$$$$$$$$$$$$ Overall Results 


#SVMR
library(vcd)
confint(Kappa(svmRadial.Conf$table))

library(MLmetrics)
F1_Score(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")
#Specificity(as.factor(svmRadial.pred), as.factor(testData$CLASS), positive = "BRCA")
#AUC(as.factor(svmRadial.pred), as.factor(testData$CLASS), positive = "BRCA")
library(epiR)
epi.tests(svmRadial.Conf$table, conf.level = 0.95)

library(pROC)
roc.multi <- multiclass.roc(as.ordered(svmRadial.pred), as.ordered(testData$CancerType))
auc(roc.multi)

svmradial_auc <- c("SVM radial Multi-class area under the curve" = auc(roc.multi))

#SVML
confint(Kappa(svmLinear.Conf$table))

library(MLmetrics)
F1_Score(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
#Specificity(as.factor(svmLinear.pred), as.factor(testData$CLASS), positive = "BRCA")
#AUC(as.factor(svmLinear.pred), as.factor(testData$CLASS), positive = "BRCA")

epi.tests(svmLinear.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(svmLinear.pred), as.ordered(testData$CancerType))
auc(roc.multi)

svmLinear_auc <- c("SVM Linear Multi-class area under the curve" = auc(roc.multi))

#SVMP
confint(Kappa(svmPoly.Conf$table))

library(MLmetrics)
F1_Score(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")
#Specificity(as.factor(svmPoly.pred), as.factor(testData$CLASS), positive = "BRCA")
#AUC(as.factor(svmPoly.pred), as.factor(testData$CLASS), positive = "BRCA")
epi.tests(svmPoly.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(svmPoly.pred), as.ordered(testData$CancerType))
auc(roc.multi)

svmploy_auc <- c("SVM Polynomial Multi-class area under the curve" = auc(roc.multi))



#ANN
confint(Kappa(ANNModel.Conf$table))

library(MLmetrics)
F1_Score(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
#Specificity(as.factor(ANNModel.pred), as.factor(testData$CLASS), positive = "BRCA")
#AUC(as.factor(ANNModel.pred), as.factor(testData$CLASS), positive = "BRCA")
epi.tests(ANNModel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(ANNModel.pred), as.ordered(testData$CancerType))
auc(roc.multi)

ann_auc <- c("ANN Multi-class area under the curve" = auc(roc.multi))

#KNN
confint(Kappa(KNNModel.Conf$table))

library(MLmetrics)
F1_Score(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
#Specificity(as.factor(KNNModel.pred), as.factor(testData$CLASS), positive = "BRCA")
#AUC(as.factor(KNNModel.pred), as.factor(testData$CLASS), positive = "BRCA")

epi.tests(KNNModel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(KNNModel.pred), as.ordered(testData$CancerType))
auc(roc.multi)

knn_auc <- c("KNN Multi-class area under the curve" = auc(roc.multi))

#Random Forest
confint(Kappa(rfmodel.Conf$table))

library(MLmetrics)
F1_Score(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
#Specificity(as.factor(rfmodel.pred), as.factor(testData$CLASS), positive = "BRCA")
#AUC(as.factor(rfmodel.pred), as.factor(testData$CLASS), positive = "BRCA")
epi.tests(rfmodel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(rfmodel.pred), as.ordered(testData$CancerType))
rf_auc <- c("random forest Multi-class area under the curve" = auc(roc.multi))

#XgBoost
confint(Kappa(boostmodel.Conf$table))
F1_Score(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(boostmodel.Conf$table, conf.level = 0.95)
roc.multi <- multiclass.roc(as.ordered(boostmodel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
xgboost_auc <- c("xgboost Multi-class area under the curve" = auc(roc.multi))

models_auc <- data.frame(model = c(names(xgboost_auc), names(rf_auc),names(knn_auc),names(ann_auc),
                                   names(svmploy_auc),names(svmLinear_auc),names(svmradial_auc)),
                         AUC = c(xgboost_auc, rf_auc,knn_auc,ann_auc,
                                 svmploy_auc,svmLinear_auc,svmradial_auc))

pls_models <- list (models_auc = models_auc, ann_metrics= valuation_kapa_ann_pls,knn_metrics = valuation_kapa_knn_pls,boost_metrics= valuation_kapa_boost_pls,
                    rf_metrics=valuation_kapa_rf_pls, svmpoly_metrics = valuation_kapa_svmploy_pls, svmlinear_metrics = valuation_kapa_svmlinear_pls,
                    svmradial_metrics = valuation_kapa_svmradial_pls,  ann = valuation_table_ann_pls, knn= valuation_table_knn_pls, svmradial= valuation_table_svmradial_pls,
                    svmLinear = valuation_table_svmlinear_pls, svmpoly = valuation_table_svmpoly_pls, rf = evaluation_table_rf_pls, boost = evaluation_table_boost_pls)

library(openxlsx)
write.xlsx(pls_models, "pls_metrics_results.xlsx", rowNames=T, overwrite = TRUE)


#===================================== SVM Radial
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	  

SVMR_pred <- predict(svmRadial.tune$finalModel, scale(testData[,-which(names(testData) %in% c("CancerType"))]), type = 'prob')
SVMR_pred <- data.frame(SVMR_pred)
colnames(SVMR_pred) <- paste0(colnames(SVMR_pred), "_pred_SVM_radial")


data <- data.frame(CancerType = testData$CancerType)

# Create dummy variables
true_label <- model.matrix(~ CancerType - 1, data = data)
library(stringr)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")


SVMR_final_df <- cbind(true_label, SVMR_pred)
SVMR_roc_res <- multiROC::multi_roc(SVMR_final_df, force_diag=T)
SVMR_pr_res <- multiROC::multi_pr(SVMR_final_df, force_diag=T)

plot_roc_df_SVMR <- multiROC::plot_roc_data(SVMR_roc_res)
plot_pr_df_SVMR <- multiROC::plot_pr_data(SVMR_pr_res)

require(ggplot2)
library(multiROC)

ggplot(plot_roc_df_SVMR, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_svmradial_roc_curve.png")


ggplot(plot_pr_df_SVMR, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_plot_pr_df_SVM_radial.png")

#===================================== SVM Linear


SVMPL_pred <- predict(svmLinear.tune$finalModel, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
SVMPL_pred <- data.frame(SVMPL_pred)
colnames(SVMPL_pred) <- paste0(colnames(SVMPL_pred), "_pred_SVM_Linear")


data <- data.frame(CancerType = testData$CancerType)
# Create dummy variables
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

data_pred <- data.frame(CancerType = SVMPL_pred$SVMPL_pred_pred_SVML)
# Create dummy variables
SVMPL_pred <- model.matrix(~ CancerType - 1, data = data)
colnames(SVMPL_pred) <- paste0(str_replace(colnames(SVMPL_pred), pattern = "CancerType", ""), "_pred_SVM_Linear")





SVMPL_final_df <- cbind(true_label, SVMPL_pred)
SVMPL_roc_res <- multi_roc(SVMPL_final_df, force_diag=T)
SVMPL_pr_res <- multi_pr(SVMPL_final_df, force_diag=T)

plot_roc_df_SVMPL <- plot_roc_data(SVMPL_roc_res)
plot_pr_df_SVML <- plot_pr_data(SVMPL_pr_res)



ggplot(plot_roc_df_SVMPL, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_svmlinear_roc_curve.png")

ggplot(plot_pr_df_SVML, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_plot_pr_df_svmlinear.png")

#===================================== SVM Polynomial

SVMP_pred <- predict(svmPoly.tune$finalModel, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
SVMP_pred <- data.frame(SVMP_pred)
colnames(SVMP_pred) <- paste0(colnames(SVMP_pred), "_pred_SVM_polynomial")


data <- data.frame(CancerType = testData$CancerType)
# Create dummy variables
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

data_pred <- data.frame(CancerType = SVMP_pred$SVMP_pred_pred_SVMP)
# Create dummy variables
SVMPL_pred <- model.matrix(~ CancerType - 1, data = data)
colnames(SVMPL_pred) <- paste0(str_replace(colnames(SVMPL_pred), pattern = "CancerType", ""), "_pred_SVM_polynomial")


SVMP_final_df <- cbind(true_label, SVMPL_pred)
SVMP_roc_res <- multi_roc(SVMP_final_df, force_diag=T)
SVMP_pr_res <- multi_pr(SVMP_final_df, force_diag=T)

plot_roc_df_SVMP <- plot_roc_data(SVMP_roc_res)
plot_pr_df_SVMP <- plot_pr_data(SVMP_pr_res)



ggplot(plot_roc_df_SVMP, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_svmpoly_roc_curve.png")


ggplot(plot_pr_df_SVMP, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_plot_pr_df_svmpoly.png")

#===================================== Artificial Neural Networks

ANN_pred <- predict(ANNModel.tune$finalModel, scale(testData[,-which(names(testData) %in% c("CancerType"))]), type = 'raw')
ANN_pred <- data.frame(ANN_pred)
colnames(ANN_pred) <- paste(colnames(ANN_pred), "_pred_ANN")



true_label <- dummies::dummy(testData$CancerType, sep = ".")
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")



ANN_final_df <- cbind(true_label, ANN_pred)
ANN_roc_res <- multi_roc(ANN_final_df, force_diag=T)
ANN_pr_res <- multi_pr(ANN_final_df, force_diag=T)

plot_roc_df_ANN <- plot_roc_data(ANN_roc_res)
plot_pr_df_ANN <- plot_pr_data(ANN_pr_res)


ggplot(plot_roc_df_ANN, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))
ggsave("pls_ann_roc_curve.png")
ggplot(plot_pr_df_ANN, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))
ggsave("pls_plot_pr_df_ann.png")

#===================================== K-nearest Nighbor

KNN_pred <- predict(KNNModel.tune$finalModel, scale(testData[,-which(names(testData) %in% c("CancerType"))]), type = 'prob')
KNN_pred <- data.frame(KNN_pred)
colnames(KNN_pred) <- paste0(colnames(KNN_pred), "_pred_KNN")


data <- data.frame(CancerType = testData$CancerType)
# Create dummy variables
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

KNN_final_df <- cbind(true_label, KNN_pred)
KNN_roc_res <- multi_roc(KNN_final_df, force_diag=T)
KNN_pr_res <- multi_pr(KNN_final_df, force_diag=T)

plot_roc_df_KNN <- plot_roc_data(KNN_roc_res)
plot_pr_df_KNN <- plot_pr_data(KNN_pr_res)

require(ggplot2)

ggplot(plot_roc_df_KNN, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))
ggsave("pls_knn_roc_curve.png")

ggplot(plot_pr_df_KNN, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_plot_pr_df_knn.png")


######Xgboost
#===================================== Xgboost

rf_pred <- predict(RFModel.tune$finalModel, scale(testData[,-which(names(testData) %in% c("CancerType"))]), type = 'prob')
rf_pred <- data.frame(rf_pred)
colnames(rf_pred) <- paste0(colnames(rf_pred), "_pred_random_forest")


data <- data.frame(CancerType = testData$CancerType)
# Create dummy variables
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")



rf_final_df <- cbind(true_label, rf_pred)
rf_roc_res <- multi_roc(rf_final_df, force_diag=T)
rf_pr_res <- multi_pr(rf_final_df, force_diag=T)

plot_roc_df_rf <- plot_roc_data(rf_roc_res)
plot_pr_df_rf <- plot_pr_data(rf_pr_res)

require(ggplot2)

ggplot(plot_roc_df_rf, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))
ggsave("pls_rf_roc_curve.png")

ggplot(plot_pr_df_rf, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_plot_pr_df_rf.png")



######Xgboost
#===================================== Xgboost

xgb_pred <- data.frame(predict(XGBModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]), type ="prob"))

xgb_pred <- t(apply(xgb_pred, MARGIN = 1, FUN = function(x){ifelse(x==max(x), 1, 0)}))

names(xgb_pred) <- unique(testData$CancerType)
colnames(xgb_pred) <- paste(colnames(xgb_pred), "_pred_xgboost")


true_label <- dummies::dummy(testData$CancerType, sep = ".")
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")


xgb_final_df <- cbind(true_label, xgb_pred)
xgb_roc_res <- multi_roc(xgb_final_df, force_diag=T)
xgb_pr_res <- multi_pr(xgb_final_df, force_diag=T)

plot_roc_df_xgb <- plot_roc_data(xgb_roc_res)
plot_pr_df_xgb <- plot_pr_data(xgb_pr_res)


ggplot(plot_roc_df_xgb, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash', size=1.3) +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1, 0), legend.position=c(.98, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))
ggsave("pls_xgb_roc_curve.png")

ggplot(plot_pr_df_xgb, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("pls_plot_pr_df_xgb.png")






