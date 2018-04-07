
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.20)


# In[14]:


my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
train_X.shape


# In[15]:


from xgboost import XGBRegressor
from sklearn.cross_validation import cross_val_score


my_model = XGBRegressor(n_estimators=1000)
scores = cross_val_score(estimator=my_model,X=train_X,y=train_y,cv=20,n_jobs=-1)
print(scores.mean())
print(scores.std())


# In[16]:


scores.mean()


# In[17]:


my_model.fit(train_X,train_y,early_stopping_rounds=5, 
            eval_set=[(test_X, test_y)], verbose=False)
my_model.score(test_X,test_y)


# In[18]:


test = pd.read_csv('test.csv')
X_test=test.select_dtypes(exclude=['object'])


# In[19]:


my_imputer = Imputer()
X_test = my_imputer.fit_transform(X_test)


# In[20]:


pre=my_model.predict(X_test)


# In[21]:


pre = pd.DataFrame(pre)


# In[22]:


X_test=pd.DataFrame(X_test)
sub=pd.DataFrame()


# In[23]:


sub['Id']=test['Id']
sub['SalePrice']=pre


# In[12]:


sub.to_csv('model1.csv',index=False)


# In[3]:


data['']

