
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().magic("config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().magic('matplotlib inline')


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


train.shape


# In[4]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# In[5]:


all_data.shape


# In[6]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[7]:


train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[8]:


all_data = pd.get_dummies(all_data)


# In[9]:


all_data = all_data.fillna(all_data.mean())


# In[10]:


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# In[11]:


'SalePrice' in  X_train.columns


# In[12]:


from sklearn.model_selection import train_test_split
X_train_train, X_train_test, y_train, y_test = train_test_split(X_train.as_matrix(), y.as_matrix(), test_size=0.20)


# In[13]:


from sklearn.cross_validation import cross_val_score
from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) 
scores = cross_val_score(estimator=model_xgb,X=X_train_train,y=y_train,cv=20,n_jobs=-1)


# In[14]:


scores.mean()
print(scores)


# In[15]:


model_xgb.fit(X_train_train, y_train)
model_xgb.score(X_train_test,y_test)


# In[16]:


X_test.shape
my_imputer = Imputer()
X_test = my_imputer.fit_transform(X_test)


# In[17]:


xgb_preds = np.expm1(model_xgb.predict(X_test))


# In[18]:


xgb_preds


# In[19]:


Id = test['Id']


# In[20]:


sub = pd.DataFrame()
sub['Id'] =Id
sub['SalePrice']=xgb_preds


# In[21]:


sub.to_csv("model2.csv",index=False)


# In[22]:


X_train_test

