#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Made by Vinay Sharma in Jupyter Notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm


# In[6]:


df = pd.read_csv('iris.data')
df.head()


# In[9]:


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])


# In[16]:


clf = svm.SVC()


# In[17]:


clf.fit(X,y)


# In[27]:


prediction = clf.predict([[3.0,1.6,4,2]])

print(prediction)
# In[ ]:




