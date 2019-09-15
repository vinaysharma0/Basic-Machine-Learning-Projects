#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing


# In[2]:


df = pd.read_excel("titanic.xls")
df.head()


# In[3]:


df.drop(['body','name'],1,inplace = True)


# In[4]:


df.convert_objects(convert_numeric=True)


# In[5]:


df.fillna(0,inplace=True)


# In[6]:


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int,df[column]))
    
    return df


# In[7]:


df = handle_non_numerical_data(df)
df.head()


# In[8]:


df.drop(['ticket'],1,inplace=True)


# In[9]:


X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])


# In[10]:


clf = KMeans(n_clusters = 2)
clf.fit(X)


# In[11]:


correct = 0


# In[12]:


for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0]==y[i]:
        correct+=1


# In[14]:


print((correct/len(X))*100)


# In[ ]:





# In[ ]:




