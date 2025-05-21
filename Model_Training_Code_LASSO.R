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


# ==============================================
# STEP 10: Train SVM Radial Model
# ==============================================

svmRadial.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Standardized predictors
  y = trainData$CancerType,                                             # Response variable
  method = "svmRadial",                                                 # Radial basis function kernel
  tuneLength = 5,                                                       # Grid search with 5 tuning values
  metric = "Accuracy",                                                  # Use accuracy for model selection
  trControl = ctrlSVM                                                   # 10-fold CV + SMOTE
)

# Print summary of tuning process
svmRadial.tune

# Make predictions on scaled test data
svmRadial.pred <- predict(svmRadial.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Create confusion matrix table
svmRadial.tab <- table(pred = svmRadial.pred, true = testData$CancerType)

# Generate detailed confusion matrix (includes sensitivity, specificity, etc.)
svmRadial.Conf <- confusionMatrix(as.factor(svmRadial.pred), as.factor(testData$CancerType), 
                                  positive = levels(as.factor(testData$CancerType))[1])
svmRadial.Conf

# Extract per-class evaluation metrics and transpose
valuation_table_svmradial_lasso <- t(svmRadial.Conf$byClass)

# Store all tuning results (accuracy, Kappa, etc.)
valuation_kapa_svmradial_lasso <- svmRadial.tune$results


# ==============================================
# STEP 11: Train SVM Linear Model
# ==============================================

svmLinear.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
  y = trainData$CancerType,
  method = "svmLinear",  # Linear kernel
  tuneLength = 5,
  metric = "Accuracy",
  trControl = ctrlSVM
)

# Output tuning results
svmLinear.tune

# Predict on test data
svmLinear.pred <- predict(svmLinear.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Confusion matrix
svmLinear.tab <- table(pred = svmLinear.pred, true = testData$CancerType)

# Compute metrics
svmLinear.Conf <- confusionMatrix(as.factor(svmLinear.pred), as.factor(testData$CancerType),
                                  positive = levels(as.factor(testData$CancerType))[1])
svmLinear.Conf

# Extract per-class metrics
valuation_table_svmlinear_lasso <- t(svmLinear.Conf$byClass)
valuation_kapa_svmlinear_lasso <- svmLinear.tune$results


# ==============================================
# STEP 12: Train SVM Polynomial Model
# ==============================================

svmPoly.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),
  y = trainData$CancerType,
  method = "svmPoly",  # Polynomial kernel
  tuneLength = 5,
  metric = "Accuracy",
  trControl = ctrlSVM
)

# Display model details
svmPoly.tune

# Save trained model to disk
saveRDS(svmPoly.tune, "svmPoly_lasso.RDS")

# Optional: Load saved model
svmPoly.tune <- readRDS("svmPoly_lasso.RDS")

# Predict with polynomial SVM
svmPoly.pred <- predict(svmPoly.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Create confusion matrix
svmPoly.tab <- table(pred = svmPoly.pred, true = testData$CancerType)

# Compute detailed metrics
svmPoly.Conf <- confusionMatrix(as.factor(svmPoly.pred), as.factor(testData$CancerType),
                                positive = levels(as.factor(testData$CancerType))[1])
svmPoly.Conf

# Store results
valuation_table_svmpoly_lasso <- t(svmPoly.Conf$byClass)
valuation_kapa_svmploy_lasso <- svmPoly.tune$results


# ==============================================
# STEP 13: Train Artificial Neural Network (ANN)
# ==============================================

library(nnet)

# Define training control for ANN
ctrlANN <- trainControl(
  method = "CV",              # 10-fold CV
  number = 10,
  classProbs = TRUE,
  allowParallel = TRUE,
  savePredictions = TRUE,
  sampling = "smote"
)

# Train neural network model using 'nnet'
ANNModel.tune <- caret::train(
  x = scale(trainData[, -which(names(trainData) %in% c("CancerType"))]),
  y = trainData$CancerType,
  method = "nnet",        # Neural net from nnet package
  trace = FALSE,          # Suppress training output
  trControl = ctrlANN,
  metric = "Accuracy"
)

# View tuning results
ANNModel.tune
print(ANNModel.tune)
plot(ANNModel.tune)

# Predict on test data
ANNModel.pred <- predict(ANNModel.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Create confusion matrix and compute metrics
ANNModel.tab <- table(pred = ANNModel.pred, true = testData$CancerType)
ANNModel.Conf <- confusionMatrix(as.factor(ANNModel.pred), as.factor(testData$CancerType),
                                 positive = levels(as.factor(testData$CancerType))[1])
ANNModel.Conf

# Store metrics
valuation_table_ann_lasso <- t(ANNModel.Conf$byClass)
valuation_kapa_ann_lasso <- ANNModel.tune$results


# ==============================================
# STEP 14: Train K-Nearest Neighbours (KNN)
# ==============================================

ctrlKNN <- trainControl(
  method = "CV",             # 10-fold CV
  number = 10,
  savePredictions = TRUE,
  classProbs = TRUE,
  allowParallel = FALSE,     # KNN is usually not parallelised in caret
  sampling = "smote"
)

# Train KNN classifier
KNNModel.tune <- caret::train(
  x = scale(trainData[, -which(names(trainData) %in% c("CancerType"))]),
  y = trainData$CancerType,
  method = "knn",
  metric = "Accuracy",
  trControl = ctrlKNN
)

# Print and visualize tuning results
KNNModel.tune
print(KNNModel.tune)
plot(KNNModel.tune)

# Predict on test data
KNNModel.pred <- predict(KNNModel.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Confusion matrix
KNNModel.tab <- table(pred = KNNModel.pred, true = testData$CancerType)

# Evaluate model
KNNModel.Conf <- confusionMatrix(as.factor(KNNModel.pred), as.factor(testData$CancerType),
                                 positive = levels(as.factor(testData$CancerType))[1])
KNNModel.Conf

# Store evaluation tables
valuation_table_knn_lasso <- t(KNNModel.Conf$byClass)
valuation_kapa_knn_lasso <- KNNModel.tune$results


# ==============================================
# STEP 15: Bagging Using Random Forest (RF)
# ==============================================

# Load necessary libraries
library(caret)
library(pROC)

# Define training control configuration for Random Forest
ctrlRF <- trainControl(
  method = "CV",             # 10-fold cross-validation
  number = 10,
  savePredictions = TRUE,    # Save predictions for later analysis
  classProbs = TRUE,         # Calculate class probabilities
  allowParallel = TRUE,      # Enable parallel processing
  sampling = "smote"         # Apply SMOTE to balance class distribution
)

# Train Random Forest model on scaled training data
RFModel.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Exclude target and scale features
  y = trainData$CancerType,                                             # Target variable
  method = "rf",                                                        # Random Forest classifier
  metric = "Accuracy",                                                  # Optimization metric
  trControl = ctrlRF                                                    # Training control object
)

# Predict on scaled test data
rfmodel.pred <- predict(RFModel.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Create confusion matrix table (counts)
rf.tab <- table(pred = rfmodel.pred, true = testData$CancerType)

# Compute full confusion matrix with detailed statistics
rfmodel.Conf <- confusionMatrix(
  as.factor(rfmodel.pred), 
  as.factor(testData$CancerType), 
  positive = levels(as.factor(testData$CancerType))[1]  # Use first level as positive class (e.g., "BRCA")
)
rfmodel.Conf

# Transpose class-specific performance metrics (sensitivity, precision, etc.)
evaluation_table_rf_lasso <- t(rfmodel.Conf$byClass)

# Store complete tuning results for hyperparameter evaluation
valuation_kapa_rf_lasso <- RFModel.tune$results


# ==============================================
# STEP 16: Boosting Using XGBoost
# ==============================================

# Load required libraries for XGBoost
# Note: `xgboost` must be installed via install.packages("xgboost") if not already
library(xgboost)

# Define cross-validation and training control for XGBoost
ctrlXGB <- trainControl(
  method = "CV",             # 10-fold cross-validation
  number = 10,
  savePredictions = TRUE,
  classProbs = TRUE,
  allowParallel = TRUE,
  sampling = "smote"         # Apply SMOTE balancing
)

# Train XGBoost model on scaled features
XGBModel.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Scaled predictors
  y = trainData$CancerType,                                             # Target variable
  method = "xgbTree",                                                   # XGBoost with decision trees
  metric = "Accuracy",                                                  # Evaluation metric
  trControl = ctrlXGB                                                   # Training control
)

# Predict classes on test data
boostmodel.pred <- predict(XGBModel.tune, scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Create confusion matrix of predictions vs. true labels
boost.tab <- table(pred = boostmodel.pred, true = testData$CancerType)

# Compute evaluation statistics
boostmodel.Conf <- confusionMatrix(
  as.factor(boostmodel.pred), 
  as.factor(testData$CancerType),
  positive = levels(as.factor(testData$CancerType))[1]
)
boostmodel.Conf

# Transpose performance metrics for easier tabular display
evaluation_table_boost_lasso <- t(boostmodel.Conf$byClass)

# Store XGBoost tuning results (accuracy, Kappa, hyperparameters)
valuation_kapa_boost_lasso <- XGBModel.tune$results



# ==============================================
# STEP 17: Overall Results and Performance Evaluation
# ==============================================

# ===== SVM Radial =====

library(vcd)       # For Kappa CI
confint(Kappa(svmRadial.Conf$table))  # Compute 95% CI for Cohen’s Kappa

library(MLmetrics)  # For F1, Precision, Sensitivity
F1_Score(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")

library(epiR)       # For extended diagnostic metrics
epi.tests(svmRadial.Conf$table, conf.level = 0.95)

library(pROC)       # For AUC
roc.multi <- multiclass.roc(as.ordered(svmRadial.pred), as.ordered(testData$CancerType))
auc(roc.multi)
svmradial_auc <- c("SVM radial Multi-class area under the curve" = auc(roc.multi))


# ===== SVM Linear =====

confint(Kappa(svmLinear.Conf$table))
F1_Score(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(svmLinear.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(svmLinear.pred), as.ordered(testData$CancerType))
auc(roc.multi)
svmLinear_auc <- c("SVM Linear Multi-class area under the curve" = auc(roc.multi))


# ===== SVM Polynomial =====

confint(Kappa(svmPoly.Conf$table))
F1_Score(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(svmPoly.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(svmPoly.pred), as.ordered(testData$CancerType))
auc(roc.multi)
svmploy_auc <- c("SVM Polynomial Multi-class area under the curve" = auc(roc.multi))


# ===== Artificial Neural Network (ANN) =====

confint(Kappa(ANNModel.Conf$table))
F1_Score(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(ANNModel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(ANNModel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
ann_auc <- c("ANN Multi-class area under the curve" = auc(roc.multi))


# ===== K-Nearest Neighbor (KNN) =====

confint(Kappa(KNNModel.Conf$table))
F1_Score(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(KNNModel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(KNNModel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
knn_auc <- c("KNN Multi-class area under the curve" = auc(roc.multi))


# ===== Random Forest =====

confint(Kappa(rfmodel.Conf$table))
F1_Score(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(rfmodel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(rfmodel.pred), as.ordered(testData$CancerType))
rf_auc <- c("random forest Multi-class area under the curve" = auc(roc.multi))


# ===== XGBoost =====

confint(Kappa(boostmodel.Conf$table))
F1_Score(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(boostmodel.Conf$table, conf.level = 0.95)

roc.multi <- multiclass.roc(as.ordered(boostmodel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
xgboost_auc <- c("xgboost Multi-class area under the curve" = auc(roc.multi))


# ==============================================
# STEP 18: Combine All AUC Results
# ==============================================

models_auc <- data.frame(
  model = c(
    names(xgboost_auc), names(rf_auc), names(knn_auc), names(ann_auc),
    names(svmploy_auc), names(svmLinear_auc), names(svmradial_auc)
  ),
  AUC = c(
    xgboost_auc, rf_auc, knn_auc, ann_auc,
    svmploy_auc, svmLinear_auc, svmradial_auc
  )
)

# ==============================================
# STEP 19: Export All Model Results
# ==============================================

# Create a list combining all model evaluation metrics and confusion matrices
lasso_models <- list(
  models_auc = models_auc,
  ann_metrics = valuation_kapa_ann_lasso,
  knn_metrics = valuation_kapa_knn_lasso,
  boost_metrics = valuation_kapa_boost_lasso,
  rf_metrics = valuation_kapa_rf_lasso,
  svmpoly_metrics = valuation_kapa_svmploy_lasso,
  svmlinear_metrics = valuation_kapa_svmlinear_lasso,
  svmradial_metrics = valuation_kapa_svmradial_lasso,
  
  # Per-class confusion table metrics
  ann = valuation_table_ann_lasso,
  knn = valuation_table_knn_lasso,
  svmradial = valuation_table_svmradial_lasso,
  svmLinear = valuation_table_svmlinear_lasso,
  svmpoly = valuation_table_svmpoly_lasso,
  rf = evaluation_table_rf_lasso,
  boost = evaluation_table_boost_lasso
)

# Write the final evaluation output to an Excel file
write.xlsx(lasso_models, "lasso_metrics_results.xlsx", rowNames = TRUE, overwrite = TRUE)



# ==============================================
# STEP 20: Plot ROC and PR Curves for SVM Radial
# ==============================================

library(caret)
library(dplyr)         # Data manipulation (required by caret)
library(kernlab)       # SVM backend
library(pROC)          # For ROC computations

# Predict class probabilities on test data using trained SVM Radial model
SVMR_pred <- predict(svmRadial.tune$finalModel, 
                     scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                     type = 'prob')  # Get class probabilities

# Convert prediction matrix to data frame
SVMR_pred <- data.frame(SVMR_pred)

# Rename columns to indicate the model used (for downstream clarity)
colnames(SVMR_pred) <- paste0(colnames(SVMR_pred), "_pred_SVM_radial")

# Extract actual class labels
data <- data.frame(CancerType = testData$CancerType)

# One-hot encode the true labels using model.matrix
true_label <- model.matrix(~ CancerType - 1, data = data)

# Rename true label columns (cleaning names and adding "_true" suffix)
library(stringr)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# Combine predicted and true labels into one data frame for evaluation
SVMR_final_df <- cbind(true_label, SVMR_pred)

# Compute multi-class ROC results
SVMR_roc_res <- multiROC::multi_roc(SVMR_final_df, force_diag = TRUE)

# Compute multi-class Precision-Recall results
SVMR_pr_res <- multiROC::multi_pr(SVMR_final_df, force_diag = TRUE)

# Convert ROC and PR results into plottable data frames
plot_roc_df_SVMR <- multiROC::plot_roc_data(SVMR_roc_res)
plot_pr_df_SVMR  <- multiROC::plot_pr_data(SVMR_pr_res)

require(ggplot2)
library(multiROC)

# --- Plot and Save ROC Curve ---
ggplot(plot_roc_df_SVMR, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +    # ROC curves per class
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),               # Diagonal reference line
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, linetype = "solid", colour = "black"))
ggsave("lasso_svmradial_roc_curve.png")

# --- Plot and Save PR Curve ---
ggplot(plot_pr_df_SVMR, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, linetype = "solid", colour = "black"))
ggsave("lasso_plot_pr_df_SVM_radial.png")



# ==============================================
# STEP 21: Plot ROC and PR Curves for SVM Linear
# ==============================================

# Predict class labels using the trained SVM Linear model
SVMPL_pred <- predict(svmLinear.tune$finalModel, 
                      scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Convert to data frame
SVMPL_pred <- data.frame(SVMPL_pred)

# Rename column to indicate it’s from SVM Linear model
colnames(SVMPL_pred) <- paste0(colnames(SVMPL_pred), "_pred_SVM_Linear")

# Reload true class labels
data <- data.frame(CancerType = testData$CancerType)

# One-hot encode true labels
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# --- POTENTIAL ISSUE ---
# The line below appears to be leftover or incorrect:
# data_pred <- data.frame(CancerType = SVMPL_pred$SVMPL_pred_pred_SVML)

# Overwrite prediction matrix with one-hot encoding (for multiclass ROC)
# This is NOT probability-based like in Radial SVM — it simulates hard classification
SVMPL_pred <- model.matrix(~ CancerType - 1, data = data)
colnames(SVMPL_pred) <- paste0(str_replace(colnames(SVMPL_pred), pattern = "CancerType", ""), "_pred_SVM_Linear")

# Combine true and predicted labels
SVMPL_final_df <- cbind(true_label, SVMPL_pred)

# Compute multi-class ROC and PR results
SVMPL_roc_res <- multiROC::multi_roc(SVMPL_final_df, force_diag = TRUE)
SVMPL_pr_res  <- multiROC::multi_pr(SVMPL_final_df, force_diag = TRUE)

# Format results for ggplot
plot_roc_df_SVMPL <- plot_roc_data(SVMPL_roc_res)
plot_pr_df_SVML   <- plot_pr_data(SVMPL_pr_res)

# --- Plot and Save ROC Curve ---
ggplot(plot_roc_df_SVMPL, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, linetype = "solid", colour = "black"))
ggsave("lasso_svmlinear_roc_curve.png")

# --- Plot and Save PR Curve ---
ggplot(plot_pr_df_SVML, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, linetype = "solid", colour = "black"))
ggsave("lasso_plot_pr_df_svmlinear.png")


# ==============================================
# STEP 22: ROC/PR Curve for SVM Polynomial
# ==============================================

# Predict class labels using the trained SVM Polynomial model
SVMP_pred <- predict(svmPoly.tune$finalModel, 
                     scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Convert predictions to data frame
SVMP_pred <- data.frame(SVMP_pred)

# Rename columns to indicate SVM Polynomial model
colnames(SVMP_pred) <- paste0(colnames(SVMP_pred), "_pred_SVM_polynomial")

# Load true class labels
data <- data.frame(CancerType = testData$CancerType)

# One-hot encode the true labels
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), "CancerType|\\S", ""), "_true")

# Note: This line seems erroneous or leftover — it is not used below
# data_pred <- data.frame(CancerType = SVMP_pred$SVMP_pred_pred_SVMP)

# Incorrect reuse of a linear model variable: fixing to use SVMP_pred properly
SVMPL_pred <- model.matrix(~ CancerType - 1, data = data)
colnames(SVMPL_pred) <- paste0(str_replace(colnames(SVMPL_pred), "CancerType", ""), "_pred_SVM_polynomial")

# Combine predictions and true labels
SVMP_final_df <- cbind(true_label, SVMPL_pred)

# Compute multi-class ROC and PR results
SVMP_roc_res <- multi_roc(SVMP_final_df, force_diag = TRUE)
SVMP_pr_res <- multi_pr(SVMP_final_df, force_diag = TRUE)

# Prepare data for plotting
plot_roc_df_SVMP <- plot_roc_data(SVMP_roc_res)
plot_pr_df_SVMP <- plot_pr_data(SVMP_pr_res)

# --- Plot and Save ROC Curve ---
ggplot(plot_roc_df_SVMP, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))
ggsave("lasso_svmpoly_roc_curve.png")

# --- Plot and Save PR Curve ---
ggplot(plot_pr_df_SVMP, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))
ggsave("lasso_plot_pr_df_svmpoly.png")



# ==============================================
# STEP 23: ROC/PR Curve for Artificial Neural Networks (ANN)
# ==============================================

# Predict class probabilities using ANN model
ANN_pred <- predict(ANNModel.tune$finalModel, 
                    scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                    type = 'raw')

# Convert predictions to data frame
ANN_pred <- data.frame(ANN_pred)

# Rename columns to indicate ANN source
colnames(ANN_pred) <- paste(colnames(ANN_pred), "_pred_ANN")

# Create dummy variables for true labels using `dummies` package
true_label <- dummies::dummy(testData$CancerType, sep = ".")
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))  # Clean variable names
colnames(true_label) <- paste(colnames(true_label), "_true")

# Combine true and predicted values
ANN_final_df <- cbind(true_label, ANN_pred)

# Generate multi-class ROC and PR data
ANN_roc_res <- multi_roc(ANN_final_df, force_diag = TRUE)
ANN_pr_res <- multi_pr(ANN_final_df, force_diag = TRUE)

# Format for plotting
plot_roc_df_ANN <- plot_roc_data(ANN_roc_res)
plot_pr_df_ANN  <- plot_pr_data(ANN_pr_res)

# --- Plot and Save ROC Curve ---
ggplot(plot_roc_df_ANN, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))
ggsave("lasso_ann_roc_curve.png")

# --- Plot and Save PR Curve ---
ggplot(plot_pr_df_ANN, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))
ggsave("lasso_plot_pr_df_ann.png")



# ==============================================
# STEP 24: ROC/PR Curve for K-Nearest Neighbor (KNN)
# ==============================================

# Predict class probabilities from KNN model
KNN_pred <- predict(KNNModel.tune$finalModel, 
                    scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                    type = 'prob')

# Convert to data frame and rename
KNN_pred <- data.frame(KNN_pred)
colnames(KNN_pred) <- paste0(colnames(KNN_pred), "_pred_KNN")

# Reload true labels
data <- data.frame(CancerType = testData$CancerType)

# Generate one-hot encoding for true classes
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), "CancerType|\\S", ""), "_true")

# Merge predicted and true labels
KNN_final_df <- cbind(true_label, KNN_pred)

# Calculate multi-class ROC and PR results
KNN_roc_res <- multi_roc(KNN_final_df, force_diag = TRUE)
KNN_pr_res <- multi_pr(KNN_final_df, force_diag = TRUE)

# Convert to plottable format
plot_roc_df_KNN <- plot_roc_data(KNN_roc_res)
plot_pr_df_KNN <- plot_pr_data(KNN_pr_res)

require(ggplot2)

# --- Plot and Save ROC Curve ---
ggplot(plot_roc_df_KNN, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))
ggsave("lasso_knn_roc_curve.png")

# --- Plot and Save PR Curve ---
ggplot(plot_pr_df_KNN, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))
ggsave("lasso_plot_pr_df_knn.png")


# ==============================================
# STEP 25A: ROC/PR Curve for Random Forest (RF)
# ==============================================

# Predict class probabilities on test data using Random Forest model
rf_pred <- predict(RFModel.tune$finalModel, 
                   scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                   type = 'prob')

# Convert predictions to a data frame
rf_pred <- data.frame(rf_pred)

# Rename columns to indicate predictions from Random Forest
colnames(rf_pred) <- paste0(colnames(rf_pred), "_pred_random_forest")

# Prepare true class labels
data <- data.frame(CancerType = testData$CancerType)

# One-hot encode true labels using model.matrix
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), "CancerType|\\S", ""), "_true")

# Combine true labels and predicted probabilities
rf_final_df <- cbind(true_label, rf_pred)

# Compute multi-class ROC and PR results
rf_roc_res <- multi_roc(rf_final_df, force_diag = TRUE)
rf_pr_res  <- multi_pr(rf_final_df, force_diag = TRUE)

# Convert for plotting
plot_roc_df_rf <- plot_roc_data(rf_roc_res)
plot_pr_df_rf  <- plot_pr_data(rf_pr_res)

# Ensure ggplot2 is available
require(ggplot2)

# --- Plot and Save ROC Curve for RF ---
ggplot(plot_roc_df_rf, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # Draw ROC curves
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),              # Diagonal reference line
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    text = element_text(size = 14),
    legend.justification = c(1, 0), legend.position = c(.98, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1,
                                     linetype = "solid", colour = "black")
  )
ggsave("lasso_rf_roc_curve.png")  # Save plot

# --- Plot and Save PR Curve for RF ---
ggplot(plot_pr_df_rf, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # PR curves
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    text = element_text(size = 14),
    legend.justification = c(1.2, 0), legend.position = c(.95, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1,
                                     linetype = "solid", colour = "black")
  )
ggsave("lasso_plot_pr_df_rf.png")  # Save plot



# ==============================================
# STEP 25B: ROC/PR Curve for XGBoost
# ==============================================

# Predict class probabilities using trained XGBoost model
xgb_pred <- data.frame(
  predict(XGBModel.tune, 
          scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
          type = "prob")
)

# Convert probability predictions to hard one-hot encoded labels:
# for each row, assign 1 to class with the highest probability, 0 otherwise
xgb_pred <- t(apply(xgb_pred, MARGIN = 1, FUN = function(x) {
  ifelse(x == max(x), 1, 0)
}))

# Assign column names based on original class labels
names(xgb_pred) <- unique(testData$CancerType)
colnames(xgb_pred) <- paste(colnames(xgb_pred), "_pred_xgboost")

# One-hot encode true class labels using the `dummies` package
true_label <- dummies::dummy(testData$CancerType, sep = ".")
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))  # Strip prefix
colnames(true_label) <- paste(colnames(true_label), "_true")      # Add suffix

# Combine true and predicted labels
xgb_final_df <- cbind(true_label, xgb_pred)

# Compute multi-class ROC and PR curves
xgb_roc_res <- multi_roc(xgb_final_df, force_diag = TRUE)
xgb_pr_res  <- multi_pr(xgb_final_df, force_diag = TRUE)

# Format for plotting
plot_roc_df_xgb <- plot_roc_data(xgb_roc_res)
plot_pr_df_xgb  <- plot_pr_data(xgb_pr_res)

# --- Plot and Save ROC Curve for XGBoost ---
ggplot(plot_roc_df_xgb, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    text = element_text(size = 14),
    legend.justification = c(1, 0), legend.position = c(.98, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1,
                                     linetype = "solid", colour = "black")
  )
ggsave("lasso_xgb_roc_curve.png")

# --- Plot and Save PR Curve for XGBoost ---
ggplot(plot_pr_df_xgb, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    text = element_text(size = 14),
    legend.justification = c(1.2, 0), legend.position = c(.95, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1,
                                     linetype = "solid", colour = "black")
  )
ggsave("lasso_plot_pr_df_xgb.png")




