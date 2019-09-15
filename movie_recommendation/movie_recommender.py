#!/usr/bin/env python
# coding: utf-8

                                       #**********************This Project was originally coded in Jupyter Notebook**************************

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


df = pd.read_csv('movie_dataset.csv')
df.head()


# In[3]:


df.columns


# In[4]:


features = ['keywords','cast','genres','director']
for feature in features:
    df[feature] = df[feature].fillna('')
    print(df[feature].head())


# In[5]:


def combine_features(row):
    try:
            return row['keywords'] + ' ' + row['cast'] + row['genres'] + row['director']
    except:
        print('error : ',row)
df['combined_feature'] = df.apply(combine_features,axis = 1) #making a column and combining all the columns vertically into it.
df['combined_feature'].head()


# In[6]:


cv = CountVectorizer()

count_matrix = cv.fit_transform(df['combined_feature'])

cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)


# In[9]:


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)
print('movie_index :',movie_index)
similar_movies =  list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
i=0
print('\n')
print("Top 10 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>=10:
        break


# In[ ]:




