
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data=pd.read_csv("Concrete_Data_Yeh.csv")


# In[4]:


data.head()


# In[8]:


data.drop("water",axis=1)


# In[9]:


data.drop("superplasticizer",axis=1)


# In[10]:


target = data['age']


# In[11]:


features=data[['cement','slag','flyash']]


# In[12]:


target.head()


# In[13]:


features.head()


# In[16]:


import seaborn as sns


# In[17]:


sns.pairplot(data=data)


# In[21]:


from sklearn.cross_validation import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,train_size=0.8,random_state=42)


# In[23]:



print("Training and testing split was successful.")


# In[24]:


print(features)


# In[25]:


from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
     score = r2_score(y_true,y_predict)
     return score
       


# In[28]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[29]:


reg_fit=regressor.fit(X_train,y_train)


# In[31]:


reg_pred=reg_fit.predict(X_test)


# In[32]:


print("regression score of :", performance_metric(y_test,reg_pred))

