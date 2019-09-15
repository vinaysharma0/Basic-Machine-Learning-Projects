#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import preprocessing , neighbors , svm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('breast-cancer-wisconsin.data')
df


# In[3]:


df.replace('?',-99999,inplace = True)


# In[4]:


df.drop(['id'],1,inplace = True)


# In[5]:


df


# In[6]:


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])


# In[7]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[13]:


clf = neighbors.KNeighborsClassifier()


# In[14]:


clf.fit(X_train,y_train)


# In[15]:


accuracy = clf.score(X_test,y_test)
print(accuracy*100)


# In[16]:


example = np.array([4,2,1,1,1,2,3,2,1])


# In[17]:


example = example.reshape(1,-1)
prediction = clf.predict(example)
print(prediction)
print(accuracy)


# In[ ]:




