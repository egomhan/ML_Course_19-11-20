#!/usr/bin/env python
# coding: utf-8

# # Project
# 
# In this project, our aim is to building a model for predicting wine qualities. Our label will be `quality` column. Do not forget, this is a Classification problem!
# 
# ## Steps
# - Read the `winequality.csv` file and describe it.
# - Make at least 4 different analysis on Exploratory Data Analysis section.
# - Pre-process the dataset to get ready for ML application. (Check missing data and handle them, can we need to do scaling or feature extraction etc.)
# - Define appropriate evaluation metric for our case (classification).
# - Train and evaluate Decision Trees and at least 2 different appropriate algorithm which you can choose from scikit-learn library.
# - Is there any overfitting and underfitting? Interpret your results and try to overcome if there is any problem in a new section.
# - Create confusion metrics for each algorithm and display Accuracy, Recall, Precision and F1-Score values.
# - Analyse and compare results of 3 algorithms.
# - Select best performing model based on evaluation metric you chose on test dataset.
# 
# 
# Good luck :)

# <h2>Samuel Egomhan</h2>

# # Data

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Read csv
data = pd.read_csv("winequality.csv")
data


# In[3]:


# Describe our data for each feature and use .info() for get information about our dataset
data.describe()
# Analyse missing values


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data.isin(['?']).sum()


# # Exploratory Data Analysis

# In[7]:


# Our label Distribution (countplot)
p = sns.countplot(data=data, x = 'quality')


# In[8]:


# Example EDA (distplot)
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.distplot(data["citric acid"])


# # Preprocessing
# 
# - Are there any duplicated values? ...   yes    ...240 are duplicate
# - Do we need to do feature scaling?...   No     ... because they are invariant to the model we are using.
# - Do we need to generate new features? .... Yes  ... a new feature was created to properly classifly the wine quality to either high (1) or Low (0)
# - Split Train and Test dataset. (0.7/0.3)

# In[9]:


# Checking for Duplicates
data.duplicated().sum()


# In[10]:


#Neew to drop the duplicate because it will add weight to the duplicated entries there-by affecting the model performance
unq_data= data.drop_duplicates()
unq_data


# In[11]:


unq_data.groupby(by="quality").count()


# In[12]:


unq_data['groupquality'] = [1 if x >= 7 else 0 for x in unq_data['quality']]


# In[13]:


unq_data


# In[14]:


unq_data['groupqualityName'] = unq_data['groupquality'].replace({0:'Low quality',  1: 'high quality' })
unq_data = pd.get_dummies(unq_data, columns = ['groupquality'])


# In[15]:


unq_data


# In[16]:


unq_data=unq_data.drop(['quality','groupquality_0', 'groupquality_1'], axis = 1)


# In[17]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[18]:


label_encoder = LabelEncoder()
unq_data["Label"] = label_encoder.fit_transform(unq_data["groupqualityName"]) 
unq_data.head()


# In[19]:


unq_data["Label"].value_counts()


# In[20]:


categories = list(label_encoder.inverse_transform([0, 1, ]))
categories


# In[21]:


clases = list(set(unq_data.groupqualityName))
unq_data.drop(["groupqualityName"], axis=1, inplace=True)


# In[22]:


unq_data


# In[51]:


X = unq_data.drop(['Label'], axis = 1)
y = unq_data['Label']


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# # ML Application
# 
# - Define models.
# - Fit models.
# - Evaluate models for both train and test dataset.
# - Generate Confusion Matrix and scores of Accuracy, Recall, Precision and F1-Score.
# - Analyse occurrence of overfitting and underfitting. If there is any of them, try to overcome it within a different section.

# In[65]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))


# In[66]:


#Feature Importance
plt.figure(figsize=(12, 8))
importance = clf.feature_importances_
sns.barplot(x=importance, y=X.columns)
plt.show()


# In[67]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
pred = clf.predict(X_test)
print(classification_report(y_test,pred))


# In[68]:


print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))


# In[69]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(categories, fontsize = 12)
ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


# ### XGBoost

# In[30]:


import xgboost as xgb


# In[70]:



model5 = xgb.XGBClassifier(max_depth=4, random_state=1)
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)
print(classification_report(y_test, y_pred5))


# In[71]:


print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))


# In[72]:



cm = confusion_matrix(y_test, y_pred5)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(categories, fontsize = 12)
ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


# In[73]:


plt.figure(figsize=(12, 8))
importance = model5.feature_importances_
sns.barplot(x=importance, y=X.columns)
plt.show()


# In[41]:


X, y = data.iloc[: , :-1], data.iloc[: , -1]


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[48]:


from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier(max_depth=7, random_state=0)
clf1.fit(X_train,y_train)
print("Accuracy of train:",clf1.score(X_train,y_train))
print("Accuracy of test:",clf1.score(X_test,y_test))


# # Evaluation
# 
# - Select the best performing model and write your comments about why choose this model.  ... Xgboost... was the best performing model because it has more of the feature importance although the acurancy for clf and model5 were the same. 
# - Analyse results and make comment about how you can improve model. # one way to improve the model is to colllect more data set for wine quality.

# In[ ]:




