# use xgboost to detect fraud
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

# load data
df = pd.read_csv('data/creditcard.csv')
print(df.head())
print(df.shape)
print(df.describe())

# split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(predictions)

# evaluate
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# plot
precision, recall, thresholds = precision_recall_curve(y_test, predictions)
auc_score = auc(recall, precision)
print('AUC: %.3f' % auc_score)
fpr, tpr, thresholds = roc_curve(y_test, predictions)
auc_score = auc(fpr, tpr)
print('AUC: %.3f' % auc_score)

# save model
model.save_model('data/xgboost.model')
print('Model saved')

# load model
model = xgb.XGBClassifier()
model.load_model('data/xgboost.model')
print('Model loaded')



