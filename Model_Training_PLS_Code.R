#===========================#
#   Set Working Directory  #
#===========================#

# For Windows users (commented out):
# setwd("C:\\Users\\John\\OneDrive\\Masters Thesis Analysis\\DataWindows\\Project")

# For Mac users (active line):
setwd("/Users/johnlunalo/Library/CloudStorage/OneDrive-Personal/Masters Thesis Analysis/DataWindows/Project")


#===========================#
#     Load and Prepare Data#
#===========================#

# Read the dataset with first column as row names and convert strings to factors
Data <- read.csv(file = "model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Recode the CLASS variable to numeric values for modeling (classification targets):
# 1 = BRCA, 2 = COAD, 3 = LUAD, 4 = OV, 5 = Other/Default
Data$CLASS <- ifelse(Data$CLASS == "BRCA", 1, 
                ifelse(Data$CLASS == "COAD", 2, 
                  ifelse(Data$CLASS == "LUAD", 3, 
                    ifelse(Data$CLASS == "OV", 4, 5))))

# Check distribution of class labels after recoding
table(Data$CLASS)


#===========================#
#     Load Required Libraries
#===========================#

library(tidyverse)   # For data manipulation and visualization
library(caret)       # For machine learning modeling and cross-validation
library(pls)         # Partial Least Squares regression
library(ggplot2)     # Advanced plotting
library(vip)         # Variable importance plots
library(chillR)      # Not used here, possibly for later weather modeling
library(mdatools)    # Multivariate data analysis tools
library(glmnet)      # For penalized regression models (like LASSO)


#===========================#
#     Data Partitioning    #
#===========================#

# Specify the name of the target variable
target <- "CLASS"

# Set seed for reproducibility
set.seed(123)

# Split the dataset into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(Data$CLASS, p = 0.70, list = FALSE)
trainData <- Data[trainIndex, ]  # Training set
testData <- Data[-trainIndex, ]  # Testing set


#===========================#
#   Enable Parallel Backend #
#===========================#

library(doParallel)

# Detect number of cores and use all except 2 (for system responsiveness)
nocores <- detectCores() - 2
cl <- makeCluster(nocores)
registerDoParallel(cl)  # Register parallel backend for caret training


#===========================#
#      Train SVM Models    #
#===========================#

library(dplyr)         # Needed for some caret backend operations
library(kernlab)       # For SVM via caret
library(pROC)	         # For AUC and ROC curve evaluations

# Define training control for caret with 10-fold cross-validation
fitControl <- trainControl(
  method = "CV",             # Cross-validation
  number = 10,               # Number of folds
  savePredictions = TRUE,    # Save predictions for performance assessment
  allowParallel = TRUE,      # Enable parallel processing
  p = 0.85,                  # Not relevant for CV; mainly used in other resampling methods
  preProc = c("scale")       # Scale predictors to mean=0 and sd=1
)


#===========================#
#   Train a PLS Model       #
#===========================#

# Train Partial Least Squares model on the full dataset
pls_fit1 <- caret::train(
  CLASS ~ ., 
  data = Data, 
  method = "pls", 
  trControl = fitControl
)


#===========================#
#   Plot Model Performance  #
#===========================#

# Plot Root Mean Squared Error (RMSE) vs. Number of PLS Components
ggplot(pls_fit1, aesthetics = aes(fill = "#0083b8")) +  # Specify fill color (though it might not apply here)
  theme_minimal() +                                    # Minimal plot theme
  xlab("Principal Components") +                       # X-axis label
  ggtitle("Scree Plot of RMSE Against Principal Components") +  # Plot title
  theme(
    plot.title = element_text(hjust = 0.5, colour = "black", size = 22),  # Centered and styled title
    axis.title = element_text(colour = "black", size = 16)               # Axis title styling
  )


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


# Load the variable importance and accuracy results from PLS modeling
df <- read.csv("vip_accuracy.csv")

############ End of selecting Variables ############

# Select the variable with the highest accuracy (VIP - Variable Importance in Projection)
selected_vip <- df[df$accuracy == max(df$accuracy), "varimp_vec_loop"][1]  # Top variable
selected_vip1 <- df[df$accuracy == max(df$accuracy), "varimp_vec_loop"][2] # Second variable with same accuracy
selected_vip2 <- df[df$accuracy == max(df$accuracy), "varimp_vec_loop"][3] # Third variable with same accuracy

# Plot accuracy against variable importance index using ggplot
ggplot(df, mapping = aes(x = varimp_vec_loop, y = accuracy)) +
  geom_line(color = "#0083b8") +  # Line plot in blue for accuracy trend
  theme_minimal() +              # Minimal theme for cleaner visualization
  xlab("Variable Importance") +  # X-axis label
  ggtitle("Accuracy against VIP") +  # Title
  theme(
    plot.title = element_text(hjust = 0.5, colour = "black", size = 22),  # Centered and styled title
    axis.title = element_text(colour = "black", size = 16)                # Axis title styling
  ) +
  # Annotate top variables on the plot with the accuracy value as a percentage
  geom_text(aes(x = selected_vip, y = max(accuracy) + 0.003, 
                label = paste0(round(max(accuracy) * 100, 2), "%")), size = 4) +
  # Second annotation commented out
  # geom_text(aes(x = selected_vip1, y = max(accuracy)+.003, label = paste0(round(max(accuracy)*100, 2), "%")), size = 4) +
  geom_text(aes(x = selected_vip2, y = max(accuracy) + 0.003, 
                label = paste0(round(max(accuracy) * 100, 2), "%")), size = 4)

# Save the plot as an image
ggsave("vip_optimum.png")

# Extract variable importance object from PLS model (assumes `pls_fit1` is a fitted PLS model)
varimp_vec <- varImp(pls_fit1, ncomp = 3)

# Count how many features have importance score greater than the selected VIP threshold
sum(varimp_vec$importance > selected_vip)

# Extract the loadings for the first two components from the PLS model
comp_loadings <- pls_fit1$finalModel$loadings[1:14899, 1:2]  # First 14,899 features across 2 components

# The following commented-out code was for examining loadings
# comp_loadings1 <- data.frame(cols = unlist(dimnames(comp_loadings)[1]))
# comp_loadings1$trial <- as.vector(apply(comp_loadings, MARGIN = 1, FUN = function(x){max(abs(x))}))

# Optional: VIP plot for top 20 features (currently commented out)
# vip::vip(pls_fit1, num_features = 20, aesthetics = aes(fill = "#0083b8")) + 
#   theme_minimal() + ylab("Importance") + 
#   ggtitle("Variable Importance for Top 20 Features") + 
#   theme(plot.title = element_text(hjust = 0.5, colour = "black", size = 22),
#         axis.title = element_text(colour = "black", size = 16)) + coord_flip()
# ggsave("vis.png")

# Extract VIP scores as a data frame
varimp_vec_df <- varimp_vec[[1]]
varimp_vec_df$var_name <- rownames(varimp_vec_df)

# Select features with overall VIP score greater than the selected threshold
selected_vars <- varimp_vec_df[varimp_vec_df$Overall > selected_vip, "var_name"]
selected_vars_pls <- selected_vars  # Store selected variables for reuse

# Optional: check overlap with LASSO-selected features (commented out)
# unique(c(selected_vars_pls[selected_vars_pls %in% row.names(LassoCof_NonZero)],
#          row.names(LassoCof_NonZero)[row.names(LassoCof_NonZero) %in% selected_vars_pls]))

### Create a dataframe of selected PLS features
selected_vars1_df <- data.frame(features = selected_vars)

# Save selected features to CSV
write.csv(x = selected_vars1_df, file = "plsselected.csv", row.names = TRUE)

############ Save data for the selected genes ############

# Load the list of selected features
plsselected <- read.csv(file = "plsselected.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Load the full dataset
Data <- read.csv(file = "model_data.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)

# Subset the dataset to only include selected features
plsData <- Data[, names(Data) %in% plsselected$features]

# Add the class label back to the new dataset
plsData$CLASS <- Data$CLASS[match(row.names(plsData), row.names(Data))]

# Final plsData now contains only selected features + CLASS for modeling


############Save Final Data from Lasso
plsData$CancerType <- plsData$CLASS
plsData <- plsData[, -which(names(plsData) =="CLASS")]

write.csv(x = plsData, file = "plsDataLatest.csv", row.names = TRUE)

##################################################################################################################

#partition the data
Data <- read.csv(file = "plsDataLatest.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)
#Data$CancerType <- ifelse(Data$CancerType == "BRCA", 1, ifelse(Data$CancerType == "COAD", 2, ifelse(Data$CancerType == "LUAD", 3, ifelse(Data$CancerType == "OV", 4, 5))))

#Split data Train = 80%/test = 20% using caret
trainIndex <- createDataPartition(Data$CancerType,p=.80,list=FALSE)
trainData <- Data[trainIndex,]
testData  <- Data[-trainIndex,]
Xtrain <- as.matrix(trainData[,-which(names(trainData) %in% c("CancerType"))])
Ytrain <- as.matrix(trainData[,which(names(trainData) %in% c("CancerType"))])
Ytrain[,1]<- as.factor(Ytrain[,1])

Xtest <- as.matrix(testData[,-which(names(testData) %in% c("CancerType"))])
Ytest <- as.matrix(testData[,which(names(testData) %in% c("CancerType"))])
Ytest[,1]<- as.factor(Ytest[,1])

###Anywhere with this line ensures reproducibility of the model results
set.seed(849)


# Train model using Support Vector Machine
#trainData$CancerType <- as.factor(ifelse(trainData$CancerType == 1, "BRCA", ifelse(trainData$CancerType == 2, "COAD", ifelse(trainData$CancerType == 3, "LUAD", ifelse(trainData$CancerType == 4, "OV", "THCA")))))
#testData$CancerType <- as.factor(ifelse(testData$CancerType == 1, "BRCA", ifelse(testData$CancerType == 2, "COAD", ifelse(testData$CancerType == 3, "LUAD", ifelse(testData$CancerType == 4, "OV", "THCA")))))
library(doParallel)
nocores <- detectCores() - 1 # Ensure parallel processing to improve processing speed
cl <- makeCluster(nocores)
registerDoParallel(cl)

# Training SVM Models
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves

# Define training control settings for caret
ctrlSVM <- caret::trainControl(
  method = "CV",                  # Use cross-validation (CV)
  number = 10,                    # Perform 10-fold cross-validation
  classProbs = TRUE,              # Enable calculation of class probabilities
  savePredictions = TRUE,         # Save out-of-fold predictions
  allowParallel = TRUE,           # Allow parallel processing if available
  sampling = "smote"              # Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
)

# Train a Support Vector Machine (SVM) model with a radial basis function (RBF) kernel
svmRadial.tune <- caret::train(
  x = as.data.frame(scale(trainData[-which(names(trainData) %in% c("CancerType"))])),  # Scale training features
  y = trainData$CancerType,                       # Target variable
  method = "svmRadial",                           # Use SVM with radial kernel
  tuneLength = 5,                                 # Try 5 different combinations of tuning parameters (sigma and cost)
  # preProc = c("center", "scale"),              # Optional preprocessing (already scaling above, so commented)
  metric = "Accuracy",                            # Use Accuracy to select the best model
  trControl = ctrlSVM                             # Use the defined training control
)

# Output the results of the radial SVM tuning process
svmRadial.tune

# This line attempts to extract the overall Accuracy from the confusion matrix (not yet defined)
svmRadial.Conf$overall[1]



# Predicting using the developed model
svmRadial.pred <- predict(svmRadial.tune, scale(testData[,-which(names(testData) %in% c("CancerType"))]))
svmRadial.tab = table(pred = svmRadial.pred, true = testData[,c("CancerType")])
#Confusion matrix best on predicted vs Actual
svmRadial.Conf = confusionMatrix(as.factor(svmRadial.pred), as.factor(testData[,c("CancerType")]), positive = levels(testData[,c("CancerType")])[1])

svmRadial.Conf

valuation_table_svmradial_pls <- t(svmRadial.Conf$byClass)

#Kappa values
valuation_kapa_svmradial_pls <- svmRadial.tune$results

### Train SVM using Linear Kernel
# Train a Support Vector Machine (SVM) model with a linear kernel using caret
svmLinear.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Scale the training features (excluding 'CancerType')
  y = trainData$CancerType,                                             # Target variable: 'CancerType'
  method = "svmLinear",                                                 # Specify linear kernel SVM method
  tuneLength = 5,                                                       # Try 5 different values of the cost (C) parameter
  # preProc = c("scale"),                                              # Scaling is already applied above; this line is commented out
  metric = "Accuracy",                                                  # Evaluation metric for model tuning
  trControl = ctrlSVM                                                  # Training control object (e.g., resampling strategy)
)

# Display the results of the SVM model tuning
svmLinear.tune

# Make predictions on the test set using the trained SVM model
svmLinear.pred <- predict(
  svmLinear.tune, 
  scale(testData[,-which(names(testData) %in% c("CancerType"))])        # Scale test features (excluding 'CancerType')
)

# Create a confusion matrix as a table (predicted vs true labels)
svmLinear.tab = table(
  pred = svmLinear.pred, 
  true = testData[,c("CancerType")]
)

# Compute detailed performance metrics (confusion matrix, sensitivity, specificity, etc.)
svmLinear.Conf = confusionMatrix(
  svmLinear.pred, 
  factor(testData[,c("CancerType")]), 
  positive = levels(factor(testData[,c("CancerType")]))[1]              # Specify the positive class (first factor level)
)

# Display the confusion matrix with detailed metrics
svmLinear.Conf

# Extract class-wise performance metrics (e.g., Sensitivity, Specificity, F1, etc.) and transpose for readability
valuation_table_svmlinear_pls <- t(svmLinear.Conf$byClass)

# Extract the grid search results for model performance across different tuning parameters (e.g., Accuracy for each C)
valuation_kapa_svmlinear_pls <- svmLinear.tune$results

# 
# Train a Support Vector Machine (SVM) model with a polynomial kernel using caret
svmPoly.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Scale the input features
  y = trainData$CancerType,                                             # Target variable: CancerType
  method = "svmPoly",                                                   # Use polynomial kernel for SVM
  tuneLength = 5,                                                       # Try 5 combinations of hyperparameters (degree, scale, and C)
  # preProc = c("scale"),                                              # Already scaled above; line commented
  metric = "Accuracy",                                                  # Metric for evaluating performance
  trControl = ctrlSVM                                                   # Training control defined earlier (e.g., 10-fold CV with SMOTE)
)

# Print the summary of model training and tuning results
svmPoly.tune

# Save the trained SVM model with polynomial kernel to an RDS file
saveRDS(svmPoly.tune, "svmPoly_pls.RDS")

# Reload the model from file (for reuse without retraining)
svmPoly.tune <- readRDS("svmPoly_pls.RDS")

# Use the trained model to make predictions on the scaled test set (excluding CancerType)
svmPoly.pred <- predict(
  svmPoly.tune, 
  scale(testData[,-which(names(testData) %in% c("CancerType"))])
)

# Create a confusion matrix as a basic table of predicted vs actual values
svmPoly.tab = table(
  pred = svmPoly.pred, 
  true = testData[,c("CancerType")]
)

# Generate a full confusion matrix with performance metrics (accuracy, sensitivity, specificity, etc.)
svmPoly.Conf = confusionMatrix(
  as.factor(svmPoly.pred),                             # Predicted labels
  as.factor(testData[,c("CancerType")]),               # True labels
  positive = levels(testData[,c("CancerType")])[1]     # Set the first class level as "positive"
)

# Print the detailed confusion matrix and associated statistics
svmPoly.Conf

# Extract class-wise metrics (e.g., Sensitivity, Specificity, F1) and transpose for readability
valuation_table_svmpoly_pls <- t(svmPoly.Conf$byClass)

# Extract overall model performance results from the training phase (Accuracy, Kappa, etc.)
valuation_kapa_svmploy_pls <- svmPoly.tune$results


#ANN
library(caret)
library(dplyr)         # Used by caret
library(nnet)       # support vector machine 
library(pROC)	       # plot the ROC curves

# Define training control for ANN using 10-fold cross-validation
ctrlANN <- trainControl(
  method = "CV",                    # Use k-fold cross-validation
  number = 10,                      # Set number of folds to 10
  classProbs = TRUE,               # Enable class probabilities for classification
  allowParallel = TRUE,            # Enable parallel processing if supported
  savePredictions = TRUE,          # Save model predictions for each resample
  sampling = "smote"               # Apply SMOTE for class imbalance handling
)

# Train an Artificial Neural Network (ANN) using the nnet package via caret
ANNModel.tune <- caret::train(
  x = scale(trainData[,-which(names(trainData) %in% c("CancerType"))]),  # Scale the training features
  y = trainData$CancerType,                                              # Target variable
  method = "nnet",                                                       # Use neural network from the `nnet` package
  trace = FALSE,                                                         # Suppress training output
  trControl = ctrlANN,                                                   # Use the above-defined training control
  # preProcess = c('scale'),                                            # Already scaled manually above
  metric = "Accuracy"                                                    # Optimize model based on Accuracy
)

# Display a summary of the trained ANN model
ANNModel.tune

# Print the model object with more detail
print(ANNModel.tune)

# Plot tuning results (e.g., decay vs. size, or accuracy performance)
plot(ANNModel.tune)

# Predict class labels on the test set using the trained ANN model
ANNModel.pred <- predict(
  ANNModel.tune, 
  scale(testData[,-which(names(testData) %in% c("CancerType"))])         # Scale test features
)

# Generate a confusion matrix as a table of predicted vs actual labels
ANNModel.tab = table(
  pred = ANNModel.pred, 
  true = testData[,c("CancerType")]
)

# Calculate full confusion matrix and classification performance metrics
ANNModel.Conf = confusionMatrix(
  as.factor(ANNModel.pred), 
  as.factor(testData[,c("CancerType")]), 
  positive = levels(testData[,c("CancerType")])[1]                       # Define the first level as the "positive" class
)

# Display the confusion matrix results and derived metrics
ANNModel.Conf

# Extract and transpose class-specific metrics (e.g., Sensitivity, Specificity, Precision, F1)
valuation_table_ann_pls <- t(ANNModel.Conf$byClass)

# Extract the model's performance across resampling (Accuracy, Kappa, etc.)
valuation_kapa_ann_pls <- ANNModel.tune$results

#KNN
library(caret)
library(pROC)

# Define training control for KNN using 10-fold cross-validation
ctrlKNN <- trainControl(
  method = "CV",                    # Use cross-validation
  number = 10,                      # 10-fold CV
  savePredictions = TRUE,          # Save predictions from each fold
  classProbs = TRUE,               # Enable class probabilities for classification
  allowParallel = FALSE,           # Do not use parallel computation (KNN is fast but memory-heavy)
  sampling = "smote"               # Apply SMOTE to balance classes during resampling
)

# Train a K-Nearest Neighbor model using caret
KNNModel.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Scale training features
  y = trainData$CancerType,                                             # Target variable
  method = "knn",                                                       # Use KNN algorithm
  metric = "Accuracy",                                                  # Optimize model for accuracy
  # preProcess = c('scale'),                                           # Commented out because manual scaling is already done
  trControl = ctrlKNN                                                   # Apply previously defined train control
)

# Display the model tuning results
KNNModel.tune

# Print the trained model object with detailed settings and results
print(KNNModel.tune)

# Plot the performance metrics (e.g., accuracy vs. number of neighbors)
plot(KNNModel.tune)

# Predict the test set using the trained KNN model
KNNModel.pred <- predict(
  KNNModel.tune, 
  scale(testData[,-which(names(testData) %in% c("CancerType"))])         # Scale test features
)

# Generate a confusion matrix as a basic table (predicted vs actual)
KNNModel.tab = table(
  pred = KNNModel.pred, 
  true = testData[,c("CancerType")]
)

# Compute a confusion matrix with performance statistics (accuracy, sensitivity, etc.)
KNNModel.Conf = confusionMatrix(
  as.factor(KNNModel.pred), 
  as.factor(testData[,c("CancerType")]), 
  positive = levels(testData[,c("CancerType")])[1]                       # Define positive class as the first level
)

# Display the confusion matrix and detailed metrics
KNNModel.Conf

# Extract class-wise performance metrics (e.g., Sensitivity, Specificity, F1) and transpose for presentation
valuation_table_knn_pls <- t(KNNModel.Conf$byClass)

# Extract KNN model's resampling results (e.g., Accuracy and Kappa for each k)
valuation_kapa_knn_pls <- KNNModel.tune$results


######Bagging using Random Forest

library(caret)
library(pROC)

# Set up training control for the Random Forest model using cross-validation
ctrlRF <- trainControl(
  method = "CV",                     # Use k-fold cross-validation
  number = 10,                       # Specify 10 folds for cross-validation
  savePredictions = TRUE,            # Save the predictions from each fold
  classProbs = TRUE,                 # Compute class probabilities for classification tasks
  allowParallel = TRUE,              # Enable parallel computation for faster processing
  sampling = "smote"                 # Apply SMOTE to address class imbalance in the training data
)

# Train a Random Forest model using the caret package
RFModel.tune <- caret::train(
  x = scale(trainData[-which(names(trainData) %in% c("CancerType"))]),  # Scale the predictors; remove the 'CancerType' column
  y = trainData$CancerType,                                             # Set the target variable
  method = "rf",                                                        # Specify the Random Forest algorithm
  metric = "Accuracy",                                                  # Optimize the model based on accuracy
  trControl = ctrlRF                                                    # Use the training control parameters defined above
)

# Make predictions on the test data using the trained Random Forest model.
# Note: The test data is scaled similarly (excluding the 'CancerType' column).
rfmodel.pred <- predict(
  RFModel.tune, 
  scale(testData[,-which(names(testData) %in% c("CancerType"))])
)

# Create a contingency table showing the predicted vs. true class labels
rf.tab = table(
  pred = rfmodel.pred, 
  true = testData[,c("CancerType")]
)

# Generate a detailed confusion matrix with performance statistics
rfmodel.Conf = confusionMatrix(
  as.factor(rfmodel.pred), 
  as.factor(testData[,c("CancerType")]), 
  positive = levels(testData[,c("CancerType")])[1]  # Define the first factor level as the positive class
)

# Display the confusion matrix and associated metrics (accuracy, sensitivity, specificity, etc.)
rfmodel.Conf

# Extract and transpose the class-specific evaluation metrics (e.g., Sensitivity, Specificity) for easier viewing
evaluation_table_rf_pls <- t(rfmodel.Conf$byClass)

# Save the resampling and tuning results (including overall accuracy and Kappa statistics) from the Random Forest model
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
# Load required package for Cohen’s Kappa statistic
library(vcd)

# Calculate confidence interval for Kappa statistic from the confusion matrix of SVM Radial
confint(Kappa(svmRadial.Conf$table))

# Load library for computing performance metrics
library(MLmetrics)

# Compute F1 Score for class "BRCA" in SVM Radial predictions
F1_Score(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")

# Compute Precision for class "BRCA"
Precision(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")

# Compute Sensitivity (Recall) for class "BRCA"
Sensitivity(as.factor(svmRadial.pred), as.factor(testData$CancerType), positive = "BRCA")

# Commented: Compute Specificity and AUC manually, but left disabled (requires binary classes)
# Specificity(as.factor(svmRadial.pred), as.factor(testData$CLASS), positive = "BRCA")
# AUC(as.factor(svmRadial.pred), as.factor(testData$CLASS), positive = "BRCA")

# Load library to compute detailed diagnostic test stats (Sensitivity, Specificity, LR+, LR-, etc.)
library(epiR)
epi.tests(svmRadial.Conf$table, conf.level = 0.95)

# Load ROC analysis library
library(pROC)

# Compute multiclass ROC and AUC for SVM Radial
roc.multi <- multiclass.roc(as.ordered(svmRadial.pred), as.ordered(testData$CancerType))
auc(roc.multi)

# Store AUC result with descriptive label
svmradial_auc <- c("SVM radial Multi-class area under the curve" = auc(roc.multi))


# --- SVM Linear ---

# Kappa CI for SVM Linear
confint(Kappa(svmLinear.Conf$table))

# Performance metrics for class "BRCA"
F1_Score(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(svmLinear.pred), as.factor(testData$CancerType), positive = "BRCA")
# Specificity and AUC are again commented out
# Specificity(as.factor(svmLinear.pred), as.factor(testData$CLASS), positive = "BRCA")
# AUC(as.factor(svmLinear.pred), as.factor(testData$CLASS), positive = "BRCA")

# Confusion matrix diagnostics
epi.tests(svmLinear.Conf$table, conf.level = 0.95)

# Multiclass ROC AUC
roc.multi <- multiclass.roc(as.ordered(svmLinear.pred), as.ordered(testData$CancerType))
auc(roc.multi)
svmLinear_auc <- c("SVM Linear Multi-class area under the curve" = auc(roc.multi))


# --- SVM Polynomial ---
# Calculate 95% confidence interval for Cohen's Kappa from the confusion matrix
confint(Kappa(svmPoly.Conf$table))

# Evaluate F1 Score for class "BRCA"
F1_Score(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")

# Evaluate precision for class "BRCA"
Precision(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")

# Evaluate sensitivity (recall) for class "BRCA"
Sensitivity(as.factor(svmPoly.pred), as.factor(testData$CancerType), positive = "BRCA")

# Specificity and binary AUC are commented out (likely not applicable for multiclass)
# Specificity(as.factor(svmPoly.pred), as.factor(testData$CLASS), positive = "BRCA")
# AUC(as.factor(svmPoly.pred), as.factor(testData$CLASS), positive = "BRCA")

# Calculate test statistics (sensitivity, specificity, predictive values) using confusion matrix
epi.tests(svmPoly.Conf$table, conf.level = 0.95)

# Compute multi-class ROC curve
roc.multi <- multiclass.roc(as.ordered(svmPoly.pred), as.ordered(testData$CancerType))

# Extract Area Under the Curve (AUC)
auc(roc.multi)

# Save SVM Polynomial model AUC
svmploy_auc <- c("SVM Polynomial Multi-class area under the curve" = auc(roc.multi))


# --- ANN ---
confint(Kappa(ANNModel.Conf$table))
F1_Score(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(ANNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
# Specificity and AUC skipped
epi.tests(ANNModel.Conf$table, conf.level = 0.95)
roc.multi <- multiclass.roc(as.ordered(ANNModel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
ann_auc <- c("ANN Multi-class area under the curve" = auc(roc.multi))


# --- KNN ---
confint(Kappa(KNNModel.Conf$table))
F1_Score(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(KNNModel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(KNNModel.Conf$table, conf.level = 0.95)
roc.multi <- multiclass.roc(as.ordered(KNNModel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
knn_auc <- c("KNN Multi-class area under the curve" = auc(roc.multi))


# --- Random Forest ---
confint(Kappa(rfmodel.Conf$table))
F1_Score(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(rfmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(rfmodel.Conf$table, conf.level = 0.95)
roc.multi <- multiclass.roc(as.ordered(rfmodel.pred), as.ordered(testData$CancerType))
rf_auc <- c("random forest Multi-class area under the curve" = auc(roc.multi))


# --- XgBoost ---
confint(Kappa(boostmodel.Conf$table))
F1_Score(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Precision(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
Sensitivity(as.factor(boostmodel.pred), as.factor(testData$CancerType), positive = "BRCA")
epi.tests(boostmodel.Conf$table, conf.level = 0.95)
roc.multi <- multiclass.roc(as.ordered(boostmodel.pred), as.ordered(testData$CancerType))
auc(roc.multi)
xgboost_auc <- c("xgboost Multi-class area under the curve" = auc(roc.multi))


# --- Consolidating AUC for All Models ---
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

# --- Consolidating all evaluation outputs into a list ---
pls_models <- list(
  models_auc = models_auc, 
  ann_metrics = valuation_kapa_ann_pls,
  knn_metrics = valuation_kapa_knn_pls,
  boost_metrics = valuation_kapa_boost_pls,
  rf_metrics = valuation_kapa_rf_pls,
  svmpoly_metrics = valuation_kapa_svmploy_pls,
  svmlinear_metrics = valuation_kapa_svmlinear_pls,
  svmradial_metrics = valuation_kapa_svmradial_pls,
  ann = valuation_table_ann_pls,
  knn = valuation_table_knn_pls,
  svmradial = valuation_table_svmradial_pls,
  svmLinear = valuation_table_svmlinear_pls,
  svmpoly = valuation_table_svmpoly_pls,
  rf = evaluation_table_rf_pls,
  boost = evaluation_table_boost_pls
)

# --- Export all results to Excel ---
library(openxlsx)
write.xlsx(pls_models, "pls_metrics_results.xlsx", rowNames = TRUE, overwrite = TRUE)


# === SVM Radial Probabilities Preparation for Output ===
library(caret)
library(dplyr)
library(kernlab)
library(pROC)

# Predict class probabilities for SVM Radial model on scaled test data
SVMR_pred <- predict(
  svmRadial.tune$finalModel,
  scale(testData[, -which(names(testData) %in% c("CancerType"))]),
  type = 'prob'
)

# Convert to dataframe for manipulation
SVMR_pred <- data.frame(SVMR_pred)

# Rename columns to indicate that predictions came from SVM Radial model
colnames(SVMR_pred) <- paste0(colnames(SVMR_pred), "_pred_SVM_radial")

# Create a new dataframe with true CancerType labels for future binding or export
data <- data.frame(CancerType = testData$CancerType)


# --- SVM Radial Evaluation and Plotting ---

# Convert true class labels to one-hot encoded dummy variables
true_label <- model.matrix(~ CancerType - 1, data = data)

# Load stringr to manipulate column names
library(stringr)

# Rename columns to indicate these are the true labels (e.g., "_true")
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# Combine true labels with predicted probabilities from SVM Radial model
SVMR_final_df <- cbind(true_label, SVMR_pred)

# Compute multi-class ROC metrics using multiROC package
SVMR_roc_res <- multiROC::multi_roc(SVMR_final_df, force_diag = TRUE)

# Compute multi-class Precision-Recall metrics
SVMR_pr_res <- multiROC::multi_pr(SVMR_final_df, force_diag = TRUE)

# Extract data in a format suitable for ggplot2 ROC plotting
plot_roc_df_SVMR <- multiROC::plot_roc_data(SVMR_roc_res)

# Extract data for ggplot2 PR curve plotting
plot_pr_df_SVMR <- multiROC::plot_pr_data(SVMR_pr_res)

# Load ggplot2 (though require is used here, it's better to use library for clarity)
require(ggplot2)
library(multiROC)

# Plot ROC curves for SVM Radial model
ggplot(plot_roc_df_SVMR, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) + # Plot ROC lines
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),  # Reference diagonal
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1,
                                         linetype = "solid", colour = "black"))

# Save ROC plot
ggsave("pls_svmradial_roc_curve.png")


# Plot Precision-Recall curve for SVM Radial
ggplot(plot_pr_df_SVMR, aes(x = Recall, y = Precision)) + 
  geom_path(aes(color = Group, linetype = Method), size = 1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(), 
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))

# Save PR plot
ggsave("pls_plot_pr_df_SVM_radial.png")


# --- SVM Linear Evaluation ---

# Predict class labels using final SVM Linear model
SVMPL_pred <- predict(svmLinear.tune$finalModel,
                      scale(testData[, -which(names(testData) %in% c("CancerType"))]))

# Convert predicted labels to dataframe
SVMPL_pred <- data.frame(SVMPL_pred)

# Rename predicted column to indicate source model
colnames(SVMPL_pred) <- paste0(colnames(SVMPL_pred), "_pred_SVM_Linear")

# Recreate data frame with true labels
data <- data.frame(CancerType = testData$CancerType)

# Generate dummy variables for true labels
true_label <- model.matrix(~ CancerType - 1, data = data)

# Rename columns for true labels
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# (Appears to be incorrect or unused: data_pred is created but not used or corrected)
# data_pred <- data.frame(CancerType = SVMPL_pred$SVMPL_pred_pred_SVML)

# Recalculate dummy variables for predicted labels (should match factor levels)
SVMPL_pred <- model.matrix(~ CancerType - 1, data = data)

# Rename predicted label columns to indicate SVM Linear source
colnames(SVMPL_pred) <- paste0(str_replace(colnames(SVMPL_pred), pattern = "CancerType", ""), "_pred_SVM_Linear")

# Combine true and predicted labels into one dataframe
SVMPL_final_df <- cbind(true_label, SVMPL_pred)

# Compute multi-class ROC
SVMPL_roc_res <- multi_roc(SVMPL_final_df, force_diag = TRUE)

# Compute multi-class PR
SVMPL_pr_res <- multi_pr(SVMPL_final_df, force_diag = TRUE)

# Prepare ROC and PR data for plotting
plot_roc_df_SVMPL <- plot_roc_data(SVMPL_roc_res)
plot_pr_df_SVML <- plot_pr_data(SVMPL_pr_res)

# ===================================== SVM Linear: ROC Curve Plot

# Plotting the ROC curve for the SVM Linear model using ggplot
ggplot(plot_roc_df_SVMPL, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +   # Draw ROC curve line for each class
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),               # Diagonal reference line (random performance)
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +                                                     # Apply clean black & white theme
  theme(
    plot.title = element_text(hjust = 0.5),                         # Center the title
    text = element_text(size = 14),                                # Base font size
    legend.justification = c(1, 0), legend.position = c(.98, .05), # Place legend at bottom-right
    legend.title = element_blank(),                                # Hide legend title
    legend.background = element_rect(fill = NULL, size = 1,        # Draw black border around legend
                                     linetype = "solid", colour = "black"))

# Save ROC plot as PNG
ggsave("pls_svmlinear_roc_curve.png")

# ===================================== SVM Linear: Precision-Recall Curve Plot

ggplot(plot_pr_df_SVML, aes(x = Recall, y = Precision)) + 
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +   # Draw PR curve
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    text = element_text(size = 14),
    legend.justification = c(1.2, 0), legend.position = c(.95, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1,
                                     linetype = "solid", colour = "black"))

# Save PR plot as PNG
ggsave("pls_plot_pr_df_svmlinear.png")


# ===================================== SVM Polynomial: Predictions and Evaluation

# Predict class labels using SVM Polynomial model
SVMP_pred <- predict(svmPoly.tune$finalModel, scale(testData[, -which(names(testData) %in% c("CancerType"))]))
SVMP_pred <- data.frame(SVMP_pred)
colnames(SVMP_pred) <- paste0(colnames(SVMP_pred), "_pred_SVM_polynomial")  # Rename predicted columns

# Create a new data frame containing true class labels
data <- data.frame(CancerType = testData$CancerType)

# Convert true labels to one-hot encoding
true_label <- model.matrix(~ CancerType - 1, data = data)
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# NOTE: this line appears to attempt using predicted values incorrectly (likely redundant)
# data_pred <- data.frame(CancerType = SVMP_pred$SVMP_pred_pred_SVMP)

# Again, dummy variables incorrectly generated from 'data' (should be from predictions)
# Fix needed here - SVMPL_pred likely a copy-paste error
SVMPL_pred <- model.matrix(~ CancerType - 1, data = data)
colnames(SVMPL_pred) <- paste0(str_replace(colnames(SVMPL_pred), pattern = "CancerType", ""), "_pred_SVM_polynomial")

# Combine true labels and predicted dummy matrix
SVMP_final_df <- cbind(true_label, SVMPL_pred)

# Compute multi-class ROC and PR metrics
SVMP_roc_res <- multi_roc(SVMP_final_df, force_diag = TRUE)
SVMP_pr_res <- multi_pr(SVMP_final_df, force_diag = TRUE)

# Prepare data for plotting
plot_roc_df_SVMP <- plot_roc_data(SVMP_roc_res)
plot_pr_df_SVMP <- plot_pr_data(SVMP_pr_res)

# ===================================== SVM Polynomial: ROC Curve Plot

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

ggsave("pls_svmpoly_roc_curve.png")

# ===================================== SVM Polynomial: Precision-Recall Curve Plot

ggplot(plot_pr_df_SVMP, aes(x = Recall, y = Precision)) + 
  geom_path(aes(color = Group, linetype = Method), size = 1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(), 
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))

ggsave("pls_plot_pr_df_svmpoly.png")


# ===================================== Artificial Neural Networks (ANN): Evaluation

# Predict class probabilities from ANN model
ANN_pred <- predict(ANNModel.tune$finalModel, 
                    scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                    type = 'raw')
ANN_pred <- data.frame(ANN_pred)

# Rename predicted columns to indicate ANN model source
colnames(ANN_pred) <- paste(colnames(ANN_pred), "_pred_ANN")

# Generate dummy variables for true class labels
true_label <- dummies::dummy(testData$CancerType, sep = ".")
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))  # Clean column names
colnames(true_label) <- paste(colnames(true_label), "_true")      # Append "_true"

# Combine true labels and predicted probabilities
ANN_final_df <- cbind(true_label, ANN_pred)

# Compute multi-class ROC and PR curves
ANN_roc_res <- multi_roc(ANN_final_df, force_diag = TRUE)
ANN_pr_res <- multi_pr(ANN_final_df, force_diag = TRUE)

# Extract ROC and PR data for plotting
plot_roc_df_ANN <- plot_roc_data(ANN_roc_res)
plot_pr_df_ANN <- plot_pr_data(ANN_pr_res)

# ===================================== ANN: ROC Curve Plot

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

ggsave("pls_ann_roc_curve.png")

# ===================================== ANN: Precision-Recall Curve Plot

ggplot(plot_pr_df_ANN, aes(x = Recall, y = Precision)) + 
  geom_path(aes(color = Group, linetype = Method), size = 1.1) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(), 
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))

ggsave("pls_plot_pr_df_ann.png")

# ================================
# K-Nearest Neighbors (KNN)
# ================================

# Predict class probabilities using the tuned KNN model on scaled test data
KNN_pred <- predict(KNNModel.tune$finalModel, 
                    scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                    type = 'prob')

# Convert predictions to a dataframe for further manipulation
KNN_pred <- data.frame(KNN_pred)

# Rename predicted columns to indicate KNN source
colnames(KNN_pred) <- paste0(colnames(KNN_pred), "_pred_KNN")

# Extract true class labels into a separate dataframe
data <- data.frame(CancerType = testData$CancerType)

# Create one-hot encoded dummy variables from true labels
true_label <- model.matrix(~ CancerType - 1, data = data)

# Rename columns to indicate they are true labels
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# Combine true labels and predicted probabilities into one dataframe
KNN_final_df <- cbind(true_label, KNN_pred)

# Compute ROC curve data for all classes
KNN_roc_res <- multi_roc(KNN_final_df, force_diag = TRUE)

# Compute precision-recall curve data for all classes
KNN_pr_res <- multi_pr(KNN_final_df, force_diag = TRUE)

# Extract data for plotting ROC curves
plot_roc_df_KNN <- plot_roc_data(KNN_roc_res)

# Extract data for plotting PR curves
plot_pr_df_KNN <- plot_pr_data(KNN_pr_res)

require(ggplot2)  # Load ggplot2 if not already loaded

# --- Plot ROC Curve for KNN ---
ggplot(plot_roc_df_KNN, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # ROC curves per class
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),             # Diagonal reference line
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))
ggsave("pls_knn_roc_curve.png")  # Save the ROC plot

# --- Plot PR Curve for KNN ---
ggplot(plot_pr_df_KNN, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # PR curves per class
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))
ggsave("pls_plot_pr_df_knn.png")  # Save the PR plot


# ================================
# Random Forest (RF)
# ================================

# Predict class probabilities using the tuned RF model
rf_pred <- predict(RFModel.tune$finalModel, 
                   scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
                   type = 'prob')

# Convert predictions to a dataframe
rf_pred <- data.frame(rf_pred)

# Rename predicted columns to indicate Random Forest source
colnames(rf_pred) <- paste0(colnames(rf_pred), "_pred_random_forest")

# Extract true labels
data <- data.frame(CancerType = testData$CancerType)

# One-hot encode the true class labels
true_label <- model.matrix(~ CancerType - 1, data = data)

# Label columns as true class indicators
colnames(true_label) <- paste0(str_replace(colnames(true_label), pattern = "CancerType|\\S", ""), "_true")

# Combine true and predicted data
rf_final_df <- cbind(true_label, rf_pred)

# Generate ROC and PR evaluation objects
rf_roc_res <- multi_roc(rf_final_df, force_diag = TRUE)
rf_pr_res <- multi_pr(rf_final_df, force_diag = TRUE)

# Extract ROC and PR plot data
plot_roc_df_rf <- plot_roc_data(rf_roc_res)
plot_pr_df_rf <- plot_pr_data(rf_pr_res)

require(ggplot2)

# --- Plot ROC Curve for Random Forest ---
ggplot(plot_roc_df_rf, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # ROC curves per class
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),             # Diagonal (baseline) line
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1, 0), legend.position = c(.98, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))
ggsave("pls_rf_roc_curve.png")  # Save the ROC plot

# --- Plot PR Curve for Random Forest ---
ggplot(plot_pr_df_rf, aes(x = Recall, y = Precision)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # PR curves per class
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 14),
        legend.justification = c(1.2, 0), legend.position = c(.95, .05),
        legend.title = element_blank(),
        legend.background = element_rect(fill = NULL, size = 1, 
                                         linetype = "solid", colour = "black"))
ggsave("pls_plot_pr_df_rf.png")  # Save the PR plot

###### XGBoost
# ================================
# Generate XGBoost Class Predictions and Evaluate
# ================================

# Predict class probabilities using the tuned XGBoost model
# The result is a matrix/data.frame where each column is the probability of a class
xgb_pred <- data.frame(
  predict(XGBModel.tune, 
          scale(testData[, -which(names(testData) %in% c("CancerType"))]), 
          type = "prob")
)

# Convert probability predictions to one-hot encoded format (hard classification)
# For each row (sample), set the maximum probability class to 1, others to 0
xgb_pred <- t(apply(xgb_pred, MARGIN = 1, FUN = function(x) {
  ifelse(x == max(x), 1, 0)
}))

# Set column names to match the class names
# (Assumes that testData$CancerType contains all unique class labels)
names(xgb_pred) <- unique(testData$CancerType)

# Append suffix to indicate XGBoost predictions
colnames(xgb_pred) <- paste(colnames(xgb_pred), "_pred_xgboost")


# ================================
# Prepare True Labels for Evaluation
# ================================

# Convert true class labels into one-hot encoded matrix using dummy encoding
true_label <- dummies::dummy(testData$CancerType, sep = ".")

# Clean column names: remove any prefix before the class label (e.g., "CancerType.BRCA" -> "BRCA")
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))

# Append suffix to indicate these are true labels
colnames(true_label) <- paste(colnames(true_label), "_true")


# ================================
# Evaluate XGBoost Model using multiROC
# ================================

# Combine predicted labels and true labels into a single dataframe
xgb_final_df <- cbind(true_label, xgb_pred)

# Compute multi-class ROC curve data
xgb_roc_res <- multi_roc(xgb_final_df, force_diag = TRUE)

# Compute multi-class Precision-Recall curve data
xgb_pr_res <- multi_pr(xgb_final_df, force_diag = TRUE)

# Extract dataframes suitable for plotting with ggplot2
plot_roc_df_xgb <- plot_roc_data(xgb_roc_res)
plot_pr_df_xgb <- plot_pr_data(xgb_pr_res)


# ================================
# Plot ROC Curve for XGBoost
# ================================

ggplot(plot_roc_df_xgb, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # Draw ROC curve lines
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),             # Diagonal reference line
               colour = 'grey', linetype = 'dotdash', size = 1.3) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5), 
    text = element_text(size = 14),
    legend.justification = c(1, 0), 
    legend.position = c(.98, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1, 
                                     linetype = "solid", colour = "black")
  )

# Save the ROC plot as PNG
ggsave("pls_xgb_roc_curve.png")


# ================================
# Plot Precision-Recall (PR) Curve for XGBoost
# ================================

ggplot(plot_pr_df_xgb, aes(x = Recall, y = Precision)) + 
  geom_path(aes(color = Group, linetype = Method), size = 1.1) +  # Draw PR curve lines
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5), 
    text = element_text(size = 14),
    legend.justification = c(1.2, 0), 
    legend.position = c(.95, .05),
    legend.title = element_blank(),
    legend.background = element_rect(fill = NULL, size = 1, 
                                     linetype = "solid", colour = "black")
  )

# Save the PR plot as PNG
ggsave("pls_plot_pr_df_xgb.png")
