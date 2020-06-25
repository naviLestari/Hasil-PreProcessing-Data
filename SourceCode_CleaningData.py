#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob


# In[47]:


Isp_review = pd.read_csv("DataCrawl_indihome_V4.csv")

Isp_review.shape  #untuk melihat Berapa baris, berapa kolom


# In[48]:


Isp_review.columns #untuk melihat terdapat kolom apa saja


# In[49]:


df = pd.DataFrame(Isp_review)
#df.head() #untuk melihat isi csv 5 baris teratas
df["Tweet"] = df['Tweet'].astype(str) #mengubah kolom tweet jadi string


# In[50]:


df['Tweet'][0] #tweet pertama dimulai dari index ke 0, jadi [0] = melihat tweet pertama 


# In[51]:


df['Tweet'] = df['Tweet'].str.replace('<.*?>', '') #hapus HTML Tag

df['Tweet'] = df['Tweet'].str.replace('\d+', '') #Hapus Angka

df['Tweet'] = df['Tweet'].str.replace('[^\w\s]', '') #Hapus Spesial Karakter

#lowercas
df['Tweet'] = df['Tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df['Tweet'][0]


# In[53]:


df['Tweet'][10]


# In[59]:


stop = stopwords.words('indonesian')
df['Tweet'] = df['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['Tweet'][10]


# In[62]:


df.tail()


# In[67]:


df.to_csv("Data_HslPreProcessing.csv") #Save Hasil PreProcessing Ke file csv baru


# In[68]:


pd.read_csv("Data_HslPreProcessing.csv") #cek Hasilnya di file csv baru


# In[ ]:




