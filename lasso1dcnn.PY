#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
# import SMOTE
from imblearn.over_sampling import SMOTE
import random
random.seed(1991)

#%%
# Set working directory
import os

os.chdir("/Users/johnlunalo/Library/CloudStorage/OneDrive-Personal/Masters Thesis Analysis/DataWindows/Project")

# Load the data
Data = pd.read_csv("lassoDataLatest.csv")
classes = Data['CancerType'].unique()

Data['CancerType'] = np.where(Data['CancerType'] == "BRCA", 0,
                              np.where(Data['CancerType'] == "COAD", 1,
                                       np.where(Data['CancerType'] == "LUAD", 2,
                                                np.where(Data['CancerType'] == "OV", 3, 4))))

X = Data.drop(columns=['CancerType', 'Unnamed: 0'])
Y = Data['CancerType']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=18062019)





# create a SMOTE object
sm = SMOTE(random_state=42)

# fit and resample the training data
X_train_oversampled, Y_train_oversampled  = sm.fit_resample(X_train, Y_train)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_oversampled)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for 1D CNN
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))




# Build the 1D CNN model
model = keras.Sequential([
    layers.Conv1D(filters=32, kernel_size=13, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    layers.MaxPooling1D(pool_size=5),
    layers.Flatten(),
    layers.Dense(units=5, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
history = model.fit(X_train_reshaped, Y_train_oversampled, epochs=20, batch_size=50, validation_split=0.15)

# Evaluate the model
Y_pred_probs = model.predict(X_test_reshaped)
Y_pred = np.argmax(Y_pred_probs, axis=1) + 1  # Convert back to original class labels
Y_test = Y_test.to_numpy()

# Create a confusion matrix




#%%
confusion1 = confusion_matrix(Y_test, Y_pred)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
confusion = confusion1[:-1:,1:6]

# Calculate the column totals
column_totals = np.sum(confusion, axis=0)

# Convert values into percentages of the column total
percentages = np.round((confusion / column_totals[np.newaxis, :]) * 100,0)
disp = ConfusionMatrixDisplay(confusion_matrix=percentages, display_labels = classes
                              )

disp.plot(cmap=plt.cm.Blues,values_format='.1f')
# %%

# Calculate metrics from the confusion matrix
true_positives = np.diag(confusion)
false_positives = np.sum(confusion, axis=0) - true_positives
false_negatives = np.sum(confusion, axis=1) - true_positives
true_negatives = np.sum(confusion) - (true_positives + false_positives + false_negatives)

# Calculate evaluation metrics
sensitivity = true_positives / (true_positives + false_negatives)
accuracy = (true_positives + true_negatives) / np.sum(confusion)
precision = true_positives / (true_positives + false_positives)
PPV = true_positives/(true_positives+false_positives)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
detection_rate = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
NPV = true_negatives / (true_negatives + false_negatives)
balanced_accuracy = (sensitivity+specificity)/2
Recall = true_positives / (true_positives + false_negatives)
Recall = true_positives/(true_positives+false_negatives)

# Print the evaluation metrics
print( "Class:       ", "BRCA", "COAD", "LUAD", "OV", 	"THCA")
print('Sensitivity:', np.round(sensitivity * 100, 2))
print('Accuracy:', np.round(accuracy * 100, 2))
print('Precision:', np.round(precision * 100, 2))
print('F1 Score:', np.round(f1_score * 100, 2))
print('Detection Rate:', np.round(detection_rate * 100, 2))
print('Specificity:', np.round(specificity * 100, 2))
print('Negative Predictive Rate:', np.round(NPV * 100, 2))
print('Positive Predictive Rate:', np.round(PPV * 100, 2))
print('Balanced Accuracy:', np.round(balanced_accuracy * 100, 2))
print('Recall:', np.round(Recall*100,2))


metrics_dict = { "category" : ["BRCA", "COAD", "LUAD", "OV", 	"THCA"],
"Sensitivity" : np.round(sensitivity * 100, 2), "Precision" : np.round(precision * 100, 2),
"F1 Score" : np.round(f1_score * 100, 2), "Detection Rate" : np.round(detection_rate * 100, 2),
"Specificity": np.round(specificity * 100, 2), "Negative Predictive Rate" : np.round(NPV * 100, 2),
"Positive Predictive Rate": np.round(PPV * 100, 2), "Balanced Accuracy":np.round(balanced_accuracy * 100, 2)}

df_1dcc_pls_metrics = pd.DataFrame(metrics_dict).transpose()

df_1dcc_pls_metrics.head()
df_1dcc_pls_metrics.to_excel("1dcnn_lasso_metrics.xlsx",
             sheet_name='lasso', engine='xlsxwriter')  

# %%
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
Y_test_binarized = label_binarize(Y_test, classes=np.unique(Y_test))
from matplotlib import pyplot as plt

fpr = {}
tpr = {}
thresh= {}
roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(Y_test_binarized[:,i], Y_pred_probs[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # plotting    
    plt.plot(fpr[i], tpr[i], linestyle='--', 
             label='%s vs Rest (AUC=%0.4f)'%(classes[i],roc_auc[i]))

plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='lower right')
plt.show()

#%%
#Show AUC Average Score
list_score = roc_auc.values()
sum(list_score)/len(list_score)

#0.9999624288511051
