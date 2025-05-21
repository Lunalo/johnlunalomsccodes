#windows
#setwd("C:\\Users\\John\\OneDrive\\Masters Thesis Analysis\\DataWindows\\Project")

#mac
setwd("/Users/johnlunalo/Library/CloudStorage/OneDrive-Personal/Masters Thesis Analysis/DataWindows/Project")



# ==============================================
# STEP 1: Read Raw Gene Expression Data
# ==============================================

# Load dataset from CSV(gene expression data + CLASS column)
Data <- read.csv(file = "model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Convert categorical class labels to numeric (for glmnet multinomial model)
Data$CLASS <- ifelse(Data$CLASS == "BRCA", 1, 
               ifelse(Data$CLASS == "COAD", 2, 
               ifelse(Data$CLASS == "LUAD", 3, 
               ifelse(Data$CLASS == "OV", 4, 5))))  # "THCA" assigned 5

# View distribution of classes
table(Data$CLASS)

# ==============================================
# STEP 2: Load Libraries
# ==============================================

library(tidyverse)
library(caret)
library(glmnet)

# ==============================================
# STEP 3: Fit Multinomial LASSO Model
# ==============================================

# Create design matrix (predictors), exclude intercept column
x <- model.matrix(CLASS ~ ., Data)[, -ncol(Data)]

# Set response variable
y <- Data[, "CLASS"]

# Set seed for reproducibility
set.seed(123)

# Perform cross-validated LASSO with multinomial outcome
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "multinomial")

# Plot cross-validated error vs. lambda
plot(cv.lasso)

# Extract lambda values:
cv.lasso$lambda.min  # Best lambda minimizing CV error
cv.lasso$lambda.1se  # Most regularized model within 1SE of the minimum

# Extract coefficient matrix at optimal lambda
coef(cv.lasso, cv.lasso$lambda.min)

# Store coefficients in object
Coef <- coef(cv.lasso, cv.lasso$lambda.min)

# Save coefficients to CSV (each class gets its own column)
write.csv(as.matrix(Coef), "Coef.csv", row.names = TRUE)

# Combine coefficient lists for all classes into a single matrix
best_alasso_coef2 <- do.call(cbind, coef(cv.lasso, cv.lasso$lambda.min))

# Save the combined coefficients
write.csv(as.matrix(best_alasso_coef2), "Coef.csv", row.names = TRUE)

# Convert coefficients to a data frame
best_alasso_coef2 <- as.data.frame(as.matrix(best_alasso_coef2))

# ==============================================
# STEP 4: Extract Non-Zero Coefficients
# ==============================================

# Read the saved coefficient matrix (excluding intercept rows)
ResultFS <- read.csv("Coef.csv", header = TRUE, stringsAsFactors = TRUE)[-c(1:2), ]

# Rename class columns
names(ResultFS)[2:ncol(ResultFS)] <- c("A1", "B1", "C1", "D1", "E1")
row.names(ResultFS) <- ResultFS$X
ResultFS <- ResultFS[, -1]  # Drop the original row-name column

# Filter features with at least one non-zero coefficient across classes
LassoCof_NonZero <- ResultFS[ResultFS$A1 != 0 | ResultFS$B1 != 0 | 
                             ResultFS$C1 != 0 | ResultFS$D1 != 0 | 
                             ResultFS$E1 != 0, ]

# Save selected features to CSV
write.csv(LassoCof_NonZero, "LassoCof_NonZero.csv", row.names = TRUE)

# ==============================================
# STEP 5: Filter Original Data by Selected Genes
# ==============================================

# Read list of selected features
LassoCoef <- read.csv("LassoCof_NonZero.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Reload original data
Data <- read.csv("model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Keep only selected features (genes)
LassoData <- Data[row.names(LassoCoef)]

# Add back CLASS variable
LassoData$CLASS <- Data$CLASS[match(row.names(LassoData), row.names(Data))]

# Rename and save the filtered dataset
LassoData$CancerType <- LassoData$CLASS
LassoData <- LassoData[, -which(names(LassoData) == "CLASS")]
write.csv(LassoData, "LassoDataLatest.csv", row.names = TRUE)

# ==============================================
# STEP 6: Partition Final Data for Modeling
# ==============================================

# Load the processed dataset with selected genes
Data <- read.csv("LassoDataLatest.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Convert class names to numeric
Data$CancerType <- ifelse(Data$CancerType == "BRCA", 1, 
                    ifelse(Data$CancerType == "COAD", 2, 
                    ifelse(Data$CancerType == "LUAD", 3, 
                    ifelse(Data$CancerType == "OV", 4, 5))))  # THCA = 5

library(glmnet)
set.seed(18062019)

# Split data into training (70%) and testing (30%)
trainIndex <- createDataPartition(Data$CancerType, p = .70, list = FALSE)
trainData <- Data[trainIndex, ]
testData  <- Data[-trainIndex, ]

# Prepare feature matrices and labels for training
Xtrain <- as.matrix(trainData[, -which(names(trainData) %in% c("CancerType"))])
Ytrain <- as.matrix(trainData[, "CancerType"])
Ytrain[, 1] <- as.factor(Ytrain[, 1])

# Prepare feature matrices and labels for testing
Xtest <- as.matrix(testData[, -which(names(testData) %in% c("CancerType"))])
Ytest <- as.matrix(testData[, "CancerType"])
Ytest[, 1] <- as.factor(Ytest[, 1])

# ==============================================
# STEP 7: Prepare for SVM Training
# ==============================================

# Reassign numeric class labels to original names (for reporting and ROC)
trainData$CancerType <- ifelse(trainData$CancerType == 1, "BRCA", 
                         ifelse(trainData$CancerType == 2, "COAD", 
                         ifelse(trainData$CancerType == 3, "LUAD", 
                         ifelse(trainData$CancerType == 4, "OV", "THCA"))))

testData$CancerType <- ifelse(testData$CancerType == 1, "BRCA", 
                        ifelse(testData$CancerType == 2, "COAD", 
                        ifelse(testData$CancerType == 3, "LUAD", 
                        ifelse(testData$CancerType == 4, "OV", "THCA"))))

# ==============================================
# STEP 8: Setup Parallel Processing for SVM Training
# ==============================================

library(doParallel)

# Detect number of cores and set up parallel cluster (excluding 1 for OS)
nocores <- detectCores() - 1
cl <- makeCluster(nocores)
registerDoParallel(cl)

# ==============================================
# STEP 9: Define Train Control Parameters for SVM
# ==============================================

library(dplyr)         # For data manipulation used in caret
library(kernlab)       # SVM algorithms (e.g., radial, linear, polynomial)
library(pROC)          # For ROC curves

# Define cross-validation and sampling strategy
ctrlSVM <- caret::trainControl(
  method = "CV",              # Cross-validation
  number = 10,                # 10-fold CV
  classProbs = TRUE,          # Compute class probabilities
  savePredictions = TRUE,     # Retain CV predictions
  allowParallel = TRUE,       # Use parallel processing
  sampling = "smote"          # Apply SMOTE for class balancing
)


svmRadial.tune <- caret::train(x=scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                               y= trainData$CancerType,
                               method = "svmRadial",   # Radial kernel
                               tuneLength = 5,					# 9 values of the cost function
                               #preProc = c("center","scale"),  # Center and scale data
                               metric="Accuracy",
                               trControl=ctrlSVM)


svmRadial.tune



svmRadial.pred <- predict(svmRadial.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
svmRadial.tab = table(pred = svmRadial.pred, true = testData[,c("CancerType")])
svmRadial.Conf = confusionMatrix(as.factor(svmRadial.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])

svmRadial.Conf

valuation_table_svmradial_lasso <- t(svmRadial.Conf$byClass)
valuation_kapa_svmradial_lasso <- svmRadial.tune$results

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

valuation_table_svmlinear_lasso <- t(svmLinear.Conf$byClass)
valuation_kapa_svmlinear_lasso <- svmLinear.tune$results
# 
svmPoly.tune <- caret::train(x=scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                             y= trainData$CancerType,
                             method = "svmPoly",
                             tuneLength = 5,
                             #preProc = c("scale"),
                             metric="Accuracy",
                             trControl=ctrlSVM)


svmPoly.tune

saveRDS(svmPoly.tune, "svmPoly_lasso.RDS")
svmPoly.tune <- readRDS( "svmPoly_lasso.RDS")

svmPoly.pred <- predict(svmPoly.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
svmPoly.tab = table(pred = svmPoly.pred, true = testData[,c("CancerType")])
svmPoly.Conf = confusionMatrix(as.factor(svmPoly.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
svmPoly.Conf

valuation_table_svmpoly_lasso <- t(svmPoly.Conf$byClass)
valuation_kapa_svmploy_lasso <- svmPoly.tune$results

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

valuation_table_ann_lasso <- t(ANNModel.Conf$byClass)
valuation_kapa_ann_lasso <- ANNModel.tune$results
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


valuation_table_knn_lasso <- t(KNNModel.Conf$byClass)
valuation_kapa_knn_lasso <- KNNModel.tune$results

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

evaluation_table_rf_lasso <- t(rfmodel.Conf$byClass)
valuation_kapa_rf_lasso <- RFModel.tune$results

#Boosting using XgBoost

#install.packages("xgboost")

library(caret)
library(pROC)
library(xgboost)

ctrlXGB <- trainControl(method = "CV",   # 10-fold cross-validation
                        number = 10,
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        allowParallel = TRUE,
                        sampling = "smote"
)

XGBModel.tune <- caret::train(x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
                              y = trainData$CancerType,
                              method = "xgbTree",   # XGBoost
                              metric = "Accuracy",
                              trControl = ctrlXGB
)

boostmodel.pred <- predict(XGBModel.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
boost.tab = table(pred = boostmodel.pred, true = testData[,c("CancerType")])
boostmodel.Conf = confusionMatrix(as.factor(boostmodel.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])
boostmodel.Conf

evaluation_table_boost_lasso <- t(boostmodel.Conf$byClass)
valuation_kapa_boost_lasso <- XGBModel.tune$results


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
                     
lasso_models <- list (models_auc = models_auc, ann_metrics= valuation_kapa_ann_lasso,knn_metrics = valuation_kapa_knn_lasso,boost_metrics= valuation_kapa_boost_lasso,
                      rf_metrics=valuation_kapa_rf_lasso, svmpoly_metrics = valuation_kapa_svmploy_lasso, svmlinear_metrics = valuation_kapa_svmlinear_lasso,
                      svmradial_metrics = valuation_kapa_svmradial_lasso,  ann = valuation_table_ann_lasso, knn= valuation_table_knn_lasso, svmradial= valuation_table_svmradial_lasso,
                      svmLinear = valuation_table_svmlinear_lasso, svmpoly = valuation_table_svmpoly_lasso, rf = evaluation_table_rf_lasso, boost = evaluation_table_boost_lasso)

write.xlsx(lasso_models, "lasso_metrics_results.xlsx", rowNames=T, overwrite = TRUE)


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

ggsave("lasso_svmradial_roc_curve.png")


ggplot(plot_pr_df_SVMR, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("lasso_plot_pr_df_SVM_radial.png")

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

ggsave("lasso_svmlinear_roc_curve.png")

ggplot(plot_pr_df_SVML, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("lasso_plot_pr_df_svmlinear.png")

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

ggsave("lasso_svmpoly_roc_curve.png")


ggplot(plot_pr_df_SVMP, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("lasso_plot_pr_df_svmpoly.png")

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
ggsave("lasso_ann_roc_curve.png")
ggplot(plot_pr_df_ANN, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))
ggsave("lasso_plot_pr_df_ann.png")

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
ggsave("lasso_knn_roc_curve.png")

ggplot(plot_pr_df_KNN, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("lasso_plot_pr_df_knn.png")


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
ggsave("lasso_rf_roc_curve.png")

ggplot(plot_pr_df_rf, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("lasso_plot_pr_df_rf.png")



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
ggsave("lasso_xgb_roc_curve.png")

ggplot(plot_pr_df_xgb, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification=c(1.2, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=1, 
                                         linetype="solid", colour ="black"))

ggsave("lasso_plot_pr_df_xgb.png")






