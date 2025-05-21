# %% IMPORTS
import os  # for interacting with the operating system (e.g., changing directories)
import random  # for setting seed in reproducibility
import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation and analysis
import matplotlib.pyplot as plt  # for plotting

from sklearn.model_selection import train_test_split  # to split dataset into train/test sets
from sklearn.preprocessing import StandardScaler, label_binarize  # for scaling and label binarization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc  # evaluation metrics

from imblearn.over_sampling import SMOTE  # to address class imbalance by oversampling minority classes
from tensorflow import keras  # high-level API for building neural networks
from tensorflow.keras import layers  # to define neural network layers

# %% SET WORKING DIRECTORY AND LOAD DATA
#os.chdir("/Users/johnlunalo/.../Project")  # change current directory to where the dataset is stored

Data = pd.read_csv("plsDataLatest.csv")  # load the dataset as a pandas DataFrame

classes = Data['CancerType'].unique()  # extract original class names for reference

# Convert categorical CancerType into numerical labels: BRCA=0, COAD=1, ..., THCA=4
Data['CancerType'] = np.where(Data['CancerType'] == "BRCA", 0,
                              np.where(Data['CancerType'] == "COAD", 1,
                                       np.where(Data['CancerType'] == "LUAD", 2,
                                                np.where(Data['CancerType'] == "OV", 3, 4))))

# %% DATA SPLITTING
X = Data.drop(columns=['CancerType', 'Unnamed: 0'])  # drop target and index columns
Y = Data['CancerType']  # target variable

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=18062019)

# %% SMOTE OVERSAMPLING
sm = SMOTE(random_state=42)  # initialize SMOTE to handle class imbalance
X_train_oversampled, Y_train_oversampled = sm.fit_resample(X_train, Y_train)  # apply oversampling only to training set

# %% STANDARD SCALING
scaler = StandardScaler()  # initialize the standard scaler
X_train_scaled = scaler.fit_transform(X_train_oversampled)  # fit and transform training data
X_test_scaled = scaler.transform(X_test)  # transform test data using the same scaler

# %% RESHAPE FOR CNN
# CNN expects 3D input: [samples, timesteps, features], so we reshape accordingly
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# %% BUILD CNN MODEL
model = keras.Sequential([  # initialize sequential model
    layers.Conv1D(filters=32, kernel_size=13, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),  # 1D convolution
    layers.MaxPooling1D(pool_size=5),  # downsample via max pooling
    layers.Flatten(),  # flatten for dense layer
    layers.Dense(units=5, activation='softmax')  # output layer with softmax activation (multiclass)
])

# Compile the model with cross-entropy loss and Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% TRAINING
history = model.fit(X_train_reshaped, Y_train_oversampled, epochs=20, batch_size=50, validation_split=0.15)  # train the model

# %% PREDICTION
Y_pred_probs = model.predict(X_test_reshaped)  # predict class probabilities
Y_pred = np.argmax(Y_pred_probs, axis=1)  # get predicted class index (argmax of softmax output)
Y_test_array = Y_test.to_numpy()  # convert pandas Series to numpy array

# %% CONFUSION MATRIX AND NORMALIZATION
conf_matrix = confusion_matrix(Y_test_array, Y_pred)  # generate confusion matrix

column_totals = np.sum(conf_matrix, axis=0)  # sum across rows per column (for normalization)
percentages = np.round((conf_matrix / column_totals[np.newaxis, :]) * 100, 2)  # normalize to percentages

# Display confusion matrix using Matplotlib
disp = ConfusionMatrixDisplay(confusion_matrix=percentages, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, values_format='.2f')  # plot with blue color map
plt.title("Normalized Confusion Matrix (%)")
plt.show()

# %% CALCULATE EVALUATION METRICS
tp = np.diag(conf_matrix)  # true positives are diagonal values
fp = np.sum(conf_matrix, axis=0) - tp  # false positives: sum of column minus TP
fn = np.sum(conf_matrix, axis=1) - tp  # false negatives: sum of row minus TP
tn = np.sum(conf_matrix) - (tp + fp + fn)  # all remaining elements are true negatives

# Metrics calculations
sensitivity = tp / (tp + fn)  # recall, true positive rate
specificity = tn / (tn + fp)  # true negative rate
accuracy = (tp + tn) / np.sum(conf_matrix)  # overall accuracy
precision = tp / (tp + fp)  # positive predictive value
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)  # harmonic mean
detection_rate = sensitivity  # same as recall
NPV = tn / (tn + fn)  # negative predictive value
PPV = precision  # positive predictive value again
balanced_accuracy = (sensitivity + specificity) / 2  # mean of TPR and TNR
recall = sensitivity  # alternate name for sensitivity

# %% DISPLAY METRICS
print("Class:        ", "BRCA", "COAD", "LUAD", "OV", "THCA")
print("Sensitivity:  ", np.round(sensitivity * 100, 2))
print("Accuracy:     ", np.round(accuracy * 100, 2))
print("Precision:    ", np.round(precision * 100, 2))
print("F1 Score:     ", np.round(f1_score * 100, 2))
print("DetectionRate:", np.round(detection_rate * 100, 2))
print("Specificity:  ", np.round(specificity * 100, 2))
print("NPV:          ", np.round(NPV * 100, 2))
print("PPV:          ", np.round(PPV * 100, 2))
print("Balanced Acc: ", np.round(balanced_accuracy * 100, 2))
print("Recall:       ", np.round(recall * 100, 2))

# %% SAVE METRICS TO EXCEL
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

df_metrics = pd.DataFrame(metrics_dict).transpose()  # convert to DataFrame and transpose for readability
df_metrics.to_excel("1dcnn_metrics_pls.xlsx", sheet_name='pls')  # export metrics to Excel

# %% ROC CURVE FOR MULTICLASS
random.seed(1992)  # seed for reproducibility

# Binarize true labels for one-vs-rest ROC curve
Y_test_binarized = label_binarize(Y_test_array, classes=[0, 1, 2, 3, 4])

# Initialize dictionaries for ROC metrics
fpr, tpr, thresh, roc_auc = {}, {}, {}, {}
n_class = len(classes)

# Compute ROC curve and AUC for each class
for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(Y_test_binarized[:, i], Y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], linestyle='-', label='%s vs Rest (AUC = %0.4f)' % (classes[i], roc_auc[i]))

# Plot ROC baseline and format
plt.plot([0, 1], [0, 1], 'b--')  # baseline line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.title('Multiclass ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# %% CALCULATE AVERAGE AUC
auc_score = np.mean(list(roc_auc.values()))  # average AUC score across all classes
print("Average AUC Score:", round(auc_score, 4))  # display AUC
