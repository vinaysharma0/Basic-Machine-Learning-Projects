
##********THIS PROGRAM WAS WRITTEN IN JUPYTER NOTEBOOK BY VINAY SHARMA *****
# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


df = pd.read_excel("titanic.xls")
df.head(2)


# In[3]:



df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
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


# In[4]:


df = handle_non_numerical_data(df)
from sklearn.preprocessing import StandardScaler


# In[5]:


scaler = StandardScaler()
scaler.fit(df)


# In[6]:


scaled_data = scaler.transform(df)


# In[7]:


scaled_data


# In[8]:


from sklearn.decomposition import PCA


# In[9]:


pca = PCA(n_components = 3)
pca.fit(scaled_data)


# In[10]:


x_pca = pca.transform(scaled_data)


# In[11]:


scaled_data.shape


# In[12]:


x_pca.shape


# In[13]:


x_pca


# In[14]:


x_pca[:,0].shape


# In[18]:


 
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca[:,0], x_pca[:,1],x_pca[:,2],c = df['survived'])
# plt.scatter(x_pca[:,0], x_pca[:,1],x_pca[:,2],c = df['survived'])
plt.show()


# In[ ]:




