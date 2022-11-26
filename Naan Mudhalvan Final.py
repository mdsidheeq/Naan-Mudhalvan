#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project
# 

# ### Team Members: Mohamed Sidheeq M, Gunaseelan C, Monishwaran C

# ## Topic : Revenue Prediction of Ice Cream Sales based on the surrounding Temperature

# Importing Libraries

# In[1]:


import numpy as np 
import pandas as pd


# Getting the Dataset

# In[2]:


df=pd.read_csv('icecream.csv')


# Performing data analysis operations

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df


# Finding the dimension of the dataframe

# In[8]:


df.ndim


# Checking if there is any null value

# In[9]:


df.isnull().sum()


# Finding the Shape of the dataframe

# In[10]:


df.shape


# ### Segregrating Dataset into X and Y

# In[11]:


X=df['Temperature']


# In[12]:


Y=df['Revenue']


# Importing the necessary Libraries for Training 

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ### Splitting into Train & Test Data

# In[16]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8)


# Dimension of X_train

# In[17]:


X.ndim


# ### Training the model

# In[18]:


model = LinearRegression()


# In[22]:


model.fit(np.array([X_train]).reshape(-1,1),Y_train)


# In[23]:


Y_pred = model.predict(np.array([X_test]).reshape(-1,1))


# ### Finding the Accuracy

# In[24]:


r2_score(Y_test,Y_pred)


# In[25]:


model.intercept_


# In[26]:


model.coef_


# In[32]:


x = float(input("Enter The Temperature to predict the revenue?"))
y = 21.42911737*x + 44.69034380219756
print("The predicted revenue for the day is $", y)


# In[ ]:




