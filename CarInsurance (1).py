#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import tree


# In[2]:


train_data = pd.read_csv(r'C:\ML\carinsurance\carInsurance_train.csv')


# In[3]:


train_data.head()


# In[4]:


train_data.info()


# In[5]:


train_data.describe()


# In[6]:


train_data.dtypes


# In[7]:


train_data.isnull().sum()


# In[8]:


sns.boxplot(x='Balance',data=train_data,palette='hls');


# In[9]:


train_data.Balance.max()


# In[10]:


train_data[train_data['Balance'] == 98417]


# In[11]:


train_data_new = train_data.drop(train_data.index[1742]);


# In[12]:


train_data_new.head()


# In[13]:


train_data_new.isnull().sum()


# In[14]:


train_data_new['Job'] = train_data_new['Job'].fillna(method ='pad')
train_data_new['Education'] = train_data_new['Education'].fillna(method ='pad')


# In[15]:


train_data_new['Communication'] = train_data_new['Communication'].fillna('none')
train_data_new['Outcome'] = train_data_new['Outcome'].fillna('none')


# In[16]:


train_data_new.isnull().sum()


# In[17]:


corr_matrix=train_data_new.corr().round(2)
plt.figure(figsize=(10,9)) 
sns.heatmap(data=corr_matrix, square=True, annot=True, linewidths=0.2)


# In[18]:


train_data_sub = ['Age','Balance','HHInsurance', 'CarLoan','NoOfContacts','DaysPassed','PrevAttempts','CarInsurance']
sns.pairplot(train_data_new[train_data_sub],hue='CarInsurance',size=1.5);


# In[19]:


train_data_new['AgeBinned'] = pd.qcut(train_data_new['Age'], 5 , labels = False)
train_data_new['BalanceBinned'] = pd.qcut(train_data_new['Balance'], 5,labels = False)


# In[20]:


train_data_new['CallStart'] = pd.to_datetime(train_data_new['CallStart'] )
train_data_new['CallEnd'] = pd.to_datetime(train_data_new['CallEnd'] )


# In[21]:


train_data_new['CallTime'] = (train_data_new['CallEnd'] - train_data_new['CallStart']).dt.total_seconds()


# In[22]:


train_data_new['CallTimeBinned'] = pd.qcut(train_data_new['CallTime'], 5,labels = False)


# In[23]:


train_data_new.drop(['Age','Balance','CallStart','CallEnd','CallTime'],axis = 1,inplace = True)


# In[24]:


train_data_new.head()


# In[25]:


Job = pd.get_dummies(data = train_data_new['Job'],prefix = "Job")


# In[26]:


Marital= pd.get_dummies(data = train_data_new['Marital'],prefix = "Marital")


# In[27]:


Education= pd.get_dummies(data = train_data_new['Education'],prefix="Education")


# In[28]:


Communication = pd.get_dummies(data = train_data_new['Communication'],prefix = "Communication")


# In[29]:


LastContactMonth = pd.get_dummies(data = train_data_new['LastContactMonth'],prefix= "LastContactMonth")
Outcome = pd.get_dummies(data = train_data_new['Outcome'],prefix = "Outcome")


# In[30]:


train_data_new.head()


# In[31]:


train_data_new.drop(['Job','Marital','Education','Communication','LastContactMonth','Outcome'],axis=1,inplace=True)


# In[32]:


train_data = pd.concat([train_data_new,Job,Marital,Education,Communication,LastContactMonth,Outcome],axis=1)


# In[33]:


train_data.columns


# In[34]:


train_data.head()


# In[35]:


X= train_data.drop(['CarInsurance'],axis=1).values
y=train_data['CarInsurance'].values


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)


# In[37]:


LR = LogisticRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)


# In[38]:


print ("Logistic Accuracy is %2.2f" % accuracy_score(y_test, y_pred))


# In[39]:


score_LR = cross_val_score(LR, X, y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_LR)


# In[40]:


print(classification_report(y_test, y_pred))


# In[41]:


cm = confusion_matrix(y_test,y_pred)


# In[42]:


print(cm)


# In[43]:


DT = tree.DecisionTreeClassifier(random_state = 0,class_weight="balanced",
    min_weight_fraction_leaf=0.01)
DT = DT.fit(X_train,y_train)
y_pred = DT.predict(X_test)


# In[44]:


print ("Decision Tree Accuracy is %2.2f" % accuracy_score(y_test, y_pred))


# In[45]:


score_DT = cross_val_score(DT, X, y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_DT)


# In[46]:


print(classification_report(y_test, y_pred))


# In[47]:


cm = confusion_matrix(y_test,y_pred)


# In[48]:


print(cm)


# In[49]:


rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced")
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


# In[50]:


print ("Random Forest Accuracy is %2.2f" % accuracy_score(y_test, rfc.predict(X_test)))


# In[51]:


score_rfc = cross_val_score(rfc, X, y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_rfc)


# In[52]:


print(classification_report(y_test, y_pred))


# In[53]:


cm = confusion_matrix(y_test,y_pred)


# In[54]:


print(cm)


# In[55]:


xgb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test) 


# In[56]:


print ("GradientBoost Accuracy= %2.2f" % accuracy_score(y_test,xgb.predict(X_test)))


# In[57]:


score_xgb = cross_val_score(xgb, X, y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_xgb)


# In[58]:


print(classification_report(y_test,y_pred))


# In[59]:


cm = confusion_matrix(y_test,y_pred)


# In[60]:


print(cm)


# In[61]:


train_data.head()


# In[62]:


X1 = train_data.drop(['CarInsurance'],axis=1).values
Y=train_data['CarInsurance'].values


# In[63]:


X1_train, X1_test, Y_train, Y_test = train_test_split(X1, Y, test_size = 0.30, random_state =100)


# In[64]:


model = LR.fit(X1_train,Y_train)


# In[65]:


y_pred = LR.predict(X1_test)


# In[66]:


print ("Logistic Accuracy is %2.2f" % accuracy_score(Y_test, y_pred))


# In[67]:


score_LR = cross_val_score(LR, X1, Y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_LR)


# In[68]:


print(classification_report(Y_test, y_pred))


# In[69]:


cm = confusion_matrix(Y_test,y_pred)


# In[70]:


print(cm)


# In[71]:


DT = tree.DecisionTreeClassifier(random_state = 0,class_weight="balanced",
    min_weight_fraction_leaf=0.01)
DT = DT.fit(X1_train,Y_train)
y_pred = DT.predict(X1_test)


# In[72]:


print ("Decision Tree Accuracy is %2.2f" % accuracy_score(Y_test, y_pred))


# In[73]:


score_DT = cross_val_score(DT, X1, Y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_DT)


# In[74]:


print(classification_report(Y_test, y_pred))


# In[75]:


cm = confusion_matrix(Y_test,y_pred)


# In[76]:


print(cm)


# In[77]:


rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced")
rfc.fit(X1_train, Y_train)
y_pred = rfc.predict(X1_test)


# In[78]:


print ("Random Forest Accuracy is %2.2f" % accuracy_score(Y_test, rfc.predict(X1_test)))


# In[79]:


score_rfc = cross_val_score(rfc, X1, Y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_rfc)


# In[80]:


print(classification_report(Y_test, y_pred))


# In[81]:


cm = confusion_matrix(Y_test,y_pred)


# In[82]:


print(cm)


# In[83]:


xgb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01)
xgb.fit(X1_train,Y_train)
y_pred = xgb.predict(X1_test) 


# In[84]:


print ("GradientBoost Accuracy= %2.2f" % accuracy_score(Y_test,xgb.predict(X1_test)))


# In[85]:


score_xgb = cross_val_score(xgb, X1, Y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_xgb)


# In[86]:


print(classification_report(Y_test,y_pred))


# In[87]:


cm = confusion_matrix(Y_test,y_pred)


# In[88]:


print(cm)


# In[ ]:




