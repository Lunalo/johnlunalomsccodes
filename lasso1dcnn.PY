#%%
# Import necessary libraries for data processing and modeling
import pandas as pd  # for data manipulation and analysis
import numpy as np  # for numerical operations
from sklearn.model_selection import train_test_split  # to split dataset into training and testing

#%%
# Import additional libraries for preprocessing, model creation, evaluation
from sklearn.preprocessing import StandardScaler  # to normalize feature values
from tensorflow import keras  # high-level API for building neural networks
from tensorflow.keras import layers  # module containing layers used in neural networks
from sklearn.metrics import confusion_matrix  # to evaluate classification performance using confusion matrix
from imblearn.over_sampling import SMOTE  # to balance the dataset by oversampling minority classes
import random  # Python’s random module

# Import required libraries for ROC computation
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

random.seed(1991)  # Set seed to ensure reproducibility of results

#%%
# Import os module for file path operations
import os

# Set working directory where data is stored
os.chdir("/Users/johnlunalo/Library/CloudStorage/OneDrive-Personal/Masters Thesis Analysis/DataWindows/Project")

#%%
# Load dataset containing LASSO-selected RNA-Seq features
Data = pd.read_csv("lassoDataLatest.csv")

# Save unique class labels (cancer types) for display later
classes = Data['CancerType'].unique()

# Convert categorical cancer types to numeric labels (0 to 4)
Data['CancerType'] = np.where(Data['CancerType'] == "BRCA", 0,
                              np.where(Data['CancerType'] == "COAD", 1,
                                       np.where(Data['CancerType'] == "LUAD", 2,
                                                np.where(Data['CancerType'] == "OV", 3, 4))))

#%%
# Separate features (X) and target variable (Y)
X = Data.drop(columns=['CancerType', 'Unnamed: 0'])  # drop label and index columns
Y = Data['CancerType']  # target labels

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=18062019)

#%%
# Create SMOTE object to balance the training data
sm = SMOTE(random_state=42)

# Apply SMOTE to training data
X_train_oversampled, Y_train_oversampled = sm.fit_resample(X_train, Y_train)

#%%
# Initialize StandardScaler for feature scaling
scaler = StandardScaler()

# Fit scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train_oversampled)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

#%%
# Reshape training data into 3D shape (samples, features, channels) required by 1D CNN
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))

# Reshape test data similarly
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

#%%
# Build 1D Convolutional Neural Network using Keras
model = keras.Sequential([
    layers.Conv1D(filters=32, kernel_size=13, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),  # convolution layer
    layers.MaxPooling1D(pool_size=5),  # max pooling to downsample
    layers.Flatten(),  # flatten output for dense layer
    layers.Dense(units=5, activation='softmax')  # output layer for 5-class classification
])

#%%
# Compile model using Adam optimizer and categorical crossentropy loss
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
# Train the model on training data (with 15% used for validation)
history = model.fit(X_train_reshaped, Y_train_oversampled, epochs=20, batch_size=50, validation_split=0.15)

#%%
# Predict class probabilities on test data
Y_pred_probs = model.predict(X_test_reshaped)

# Convert probabilities to predicted class labels
Y_pred = np.argmax(Y_pred_probs, axis=1) + 1  # NOTE: `+1` is likely incorrect unless label encoding starts at 1

# Convert Y_test to NumPy array for evaluation
Y_test = Y_test.to_numpy()

#%%
# Generate confusion matrix
confusion1 = confusion_matrix(Y_test, Y_pred)

#%%
# Import display module and plotting library
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Optionally adjust confusion matrix dimensions
confusion = confusion1[:-1:, 1:6]

# Calculate column-wise totals
column_totals = np.sum(confusion, axis=0)

# Normalize confusion matrix as percentage by column
percentages = np.round((confusion / column_totals[np.newaxis, :]) * 100, 0)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=percentages, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, values_format='.1f')

#%%
# Extract performance metrics from the confusion matrix
true_positives = np.diag(confusion)
false_positives = np.sum(confusion, axis=0) - true_positives
false_negatives = np.sum(confusion, axis=1) - true_positives
true_negatives = np.sum(confusion) - (true_positives + false_positives + false_negatives)

#%%
# Compute various evaluation metrics
sensitivity = true_positives / (true_positives + false_negatives)
accuracy = (true_positives + true_negatives) / np.sum(confusion)
precision = true_positives / (true_positives + false_positives)
PPV = true_positives / (true_positives + false_positives)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
detection_rate = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
NPV = true_negatives / (true_negatives + false_negatives)
balanced_accuracy = (sensitivity + specificity) / 2
Recall = true_positives / (true_positives + false_negatives)

#%%
# Print computed metrics for all cancer classes
print("Class:       ", "BRCA", "COAD", "LUAD", "OV", "THCA")
print('Sensitivity:', np.round(sensitivity * 100, 2))
print('Accuracy:', np.round(accuracy * 100, 2))
print('Precision:', np.round(precision * 100, 2))
print('F1 Score:', np.round(f1_score * 100, 2))
print('Detection Rate:', np.round(detection_rate * 100, 2))
print('Specificity:', np.round(specificity * 100, 2))
print('Negative Predictive Rate:', np.round(NPV * 100, 2))
print('Positive Predictive Rate:', np.round(PPV * 100, 2))
print('Balanced Accuracy:', np.round(balanced_accuracy * 100, 2))
print('Recall:', np.round(Recall * 100, 2))

#%%
# Construct a dictionary of metrics for DataFrame creation
metrics_dict = {
    "category": ["BRCA", "COAD", "LUAD", "OV", "THCA"],
    "Sensitivity": np.round(sensitivity * 100, 2),
    "Precision": np.round(precision * 100, 2),
    "F1 Score": np.round(f1_score * 100, 2),
    "Detection Rate": np.round(detection_rate * 100, 2),
    "Specificity": np.round(specificity * 100, 2),
    "Negative Predictive Rate": np.round(NPV * 100, 2),
    "Positive Predictive Rate": np.round(PPV * 100, 2),
    "Balanced Accuracy": np.round(balanced_accuracy * 100, 2)
}

# Create a DataFrame from the metrics dictionary
df_1dcc_pls_metrics = pd.DataFrame(metrics_dict).transpose()

# Preview metrics table
df_1dcc_pls_metrics.head()

# Export metrics to Excel file for reporting
df_1dcc_pls_metrics.to_excel("1dcnn_lasso_metrics.xlsx", sheet_name='lasso', engine='xlsxwriter')
# Binarize test labels for ROC analysis (One-vs-Rest)
Y_test_binarized = label_binarize(Y_test, classes=np.unique(Y_test))

# Initialize containers for ROC metrics
fpr = {}  # false positive rate
tpr = {}  # true positive rate
thresh = {}  # thresholds
roc_auc = dict()  # area under curve

# Total number of classes
n_class = classes.shape[0]

# Loop through each class to compute ROC curves
for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(Y_test_binarized[:, i], Y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curve for each class
    plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.4f)' % (classes[i], roc_auc[i]))

# Add diagonal reference line (random classifier)
plt.plot([0, 1], [0, 1], 'b--')

# Set plot limits and labels
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Display the ROC plot
plt.show()

#%%
# Compute and display average AUC across all classes
list_score = roc_auc.values()
sum(list_score) / len(list_score)  # average AUC
