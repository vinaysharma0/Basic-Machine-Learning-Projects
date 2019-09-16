#!/usr/bin/env python
# coding: utf-8

# In[44]:


# Made by Vinay Sharma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn import svm , neighbors


# In[109]:


df = pd.read_csv('iris.data')
df.tail()


# In[9]:


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[114]:


clf = neighbors.KNeighborsClassifier()
clf2 = svm.SVC()


# In[115]:


clf2.fit(X_train,y_train)
clf.fit(X_train,y_train)


# In[118]:


a = clf.predict([[5.9,3.0,5.1,1.8]])
b = clf2.predict([[5.9,3.0,5.1,1.8]])


# In[126]:


print(a)
print(b)
accuracy = clf.score(X_test,y_test)


# In[127]:


print(accuracy)

