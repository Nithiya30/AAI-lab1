import pandas as pd import numpy as np
df = pd.read_csv('heart.csv') # Prep features and target
X = df.drop(columns=['output']) y = df['output']
print(X.shape) print(y.shape)
#print unique values count of y class labels unique_values = np.unique(y)
counts = np.zeros_like(unique_values) for i, value in enumerate(unique_values):
counts[i] = np.count_nonzero(y == value) for i, value in enumerate(unique_values):
print(f"Class label: {value}, Count: {counts[i]}") from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #print shape of train and test data (X,y)
print(X_train.shape) print(X_test.shape) print(y_train.shape) print(y_test.shape)
# print count of o and 1 in test data unique_values = np.unique(y_test) counts = np.zeros_like(unique_values) for i, value in enumerate(unique_values):
counts[i] = np.count_nonzero(y_test == value) for i, value in enumerate(unique_values):
print(f"Class label: {value}, Count: {counts[i]}") from sklearn.preprocessing import MinMaxScaler scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test) import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score from sklearn.metrics import precision_score from sklearn.metrics import recall_score from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report from sklearn.metrics import ConfusionMatrixDisplay
 
from sklearn.metrics import precision_recall_curve,roc_curve, auc, roc_auc_score from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression logistic_model = LogisticRegression(random_state=42) logistic_model.fit(X_train, y_train)
preds = logistic_model.predict(X_test) #print confusion matrix
classes = ["Normal","AbNormal"] cm = confusion_matrix(y_test,preds) print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes) fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix") disp = disp.plot(ax=ax) plt.show()
Accuracy = metrics.accuracy_score(y_test, preds) print('Accuracy:', Accuracy*100)
Precision = metrics.precision_score(y_test, preds) print('Precision:', Precision*100)
Recall = metrics.recall_score(y_test, preds) print('Recall:', Recall*100)
F1_score = metrics.f1_score(y_test, preds) print('F1 Score:', F1_score*100)
roc_auc = roc_auc_score(y_test,preds) print('ROC AUC score: %.3f' % roc_auc) print('Mean ROC AUC: %.5f' % roc_auc.mean())
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,preds) auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--' ,'No Skill')
plt.plot(fpr_keras, tpr_keras, label = 'area = {:.3f}'.format(auc_keras)) plt.xlabel('False positive rate')
plt.ylabel('True positive rate') plt.title('ROC curve') plt.legend(loc = 'best') plt.show()
target_names=['Normal', 'Abnormal']
print (classification_report(preds, y_test,target_names=target_names))
v
