#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import math
import quandl
from sklearn import preprocessing , svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')
import seaborn as sns
sns.set()


# In[79]:


df = quandl.get('WIKI/GOOGL')
df.head()


# In[80]:


df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]


# In[81]:


df.head()


# In[82]:


df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100


# In[83]:


new = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df['PCT_CHANGE'] = new


# In[84]:


print(df.head())


# In[85]:


forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True)


# In[86]:


forecast_out = int(math.ceil(0.1*len(df)))


# In[87]:


forecast_out


# In[88]:


df['label'] = df[[forecast_col]].shift(-forecast_out)


# In[12]:


df.describe()


# In[89]:


df.dropna(inplace = True)
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])


# In[90]:


X = preprocessing.scale(X)
# X = X[:-forecast_out]
X_lately = X[-forecast_out:]


# In[68]:


X


# In[16]:


y


# In[91]:


print(len(X),len(y))


# In[92]:


X_train , X_test , y_train , y_test  = train_test_split(X,y,test_size = 0.2)


# In[93]:


clf = LinearRegression()


# In[20]:


# X_train = X_train[~np.isnan(X_train)]
# y_train = y_train[~np.isnan(y_train)]


# In[94]:


clf.fit(X_train,y_train)


# In[95]:


accuracy = clf.score(X_test,y_test)


# In[97]:


print(accuracy*100)


# In[98]:


forecast_set = clf.predict(X_lately)


# In[99]:


print(forecast_set,accuracy,forecast_out)


# In[100]:


df['forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


# In[56]:


next_unix


# In[101]:


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


# In[102]:


df.tail()


# In[104]:


df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc = 'best')
plt.ylabel('Price')
plt.show()

print(len(df))