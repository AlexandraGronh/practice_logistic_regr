#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd


# In[2]:


pima=pd.read_csv("/Users/sudipmajumder/Downloads/diabetes.csv")


# In[3]:


pima.head()


# In[5]:





# In[11]:


X=pima[['BloodPressure','Insulin','BMI','Age']]


# In[12]:


y=pima.Outcome


# In[13]:


X


# In[14]:


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)
#
y_pred=logreg.predict(X_test)


# In[17]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[18]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[21]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




