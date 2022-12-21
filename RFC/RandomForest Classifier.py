#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,  precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Importing names for the columns
ocr_names = pd.read_excel('letter.names.xlsx', header=None)
ocr_names_list = ocr_names[0].values.tolist()
# Importing the dataset and deleting non-important columns
ocr_data = pd.read_excel('letter.data.xlsx', header=None, names=ocr_names_list)
ocr_data = ocr_data.drop(columns=['id','next_id','word_id','position','fold'])

# Creating variables and dividing the set into training and testing
X, y = ocr_data.iloc[:,1:].to_numpy(), ocr_data.iloc[:,0].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=25)

# Oversampling the traning set to correct problems of unbalanced data
over_sampler = RandomOverSampler(random_state=42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)

# Fitting the model
forest_clf = RandomForestClassifier(n_estimators=500, random_state=25)
forest_clf.fit(X_res, y_res)


y_res_pred = cross_val_predict(forest_clf, X_res, y_res, cv=3)
accuracy_score(y_res, y_res_pred)


y_pred = forest_clf.predict(X_test)
accuracy_score(y_test, y_pred)


balanced_accuracy_score(y_test, y_pred)

