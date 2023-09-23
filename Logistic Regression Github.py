#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


df = pd.read_csv("titanic-data.csv")

# preview train data
df.head()


# In[2]:


df.shape #dimensions of the data


# In[3]:


df.describe() #Summarise the data


# In[4]:


df.isnull().sum() #check for any missing/null values


# Since there are missing values in the dataset, I will replace the missing values as follows:
# 1. Age will be replaced by the mean of Age
# 2. Cabin column will be dropped from the dataset completely since every passenger will have their unique cabins and thus, it would be hard to impute individual cabins
# 3. The 2 missing rows for Embarked will be dropped
# 

# In[5]:


df['Age'] = df['Age'].fillna(np.mean(df['Age']))
df = df.dropna(subset=['Embarked'])
df = df.drop('Cabin', axis = 1)
df.isnull().sum()


# Next, the columns which are discrete to each passenger will be removed as they do not contribute towards the significance of their survival rate

# In[6]:


df = df.drop(columns = ['Name','PassengerId','Ticket'], axis = 1)


# Now, to convert the 'Sex' and 'Embarked' varibles into numerical format, I will use the 'map' function to transform text to numbers

# In[7]:


df['Sex'] = df['Sex'].map({'female':1, 'male':0})
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})


# In[8]:


sns.heatmap(df.corr(), annot = True)


# From the correlation heatmap observed above, it can be inferred that Age, Siblings/Spouses Aboard, Parents/Children Aboard, Fare and, location of aboarding the ship do not contribute significantly towards the survival of the passengers.

# In[9]:


df = df.drop(columns = ['Parch','SibSp','Age'], axis = 1)
df


# In[10]:


df.shape


# I will now construct some data visualisations to derive some quick insights about the data

# In[11]:


sns.boxplot(df['Fare'])


# In[12]:


df = df[df['Fare'] <= 80] #drop the records with outliers in the fare above 80 for more accurate predictions and fitting


# In[13]:


sns.countplot(x='Survived', data=df)


# Since the target variable 'Survived' is a binary variable, I will implement the logistic regression model to the dataset.

# In[14]:


X= df.iloc[:, 1:5].values
X


# In[15]:


y = df.iloc[:, 0].values
y


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[17]:


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
logit = logit.fit(X_train, y_train)


# In[18]:


y_pred = logit.predict(X_test)
print(y_pred)


# In[19]:


print(logit.intercept_)


# In[20]:


print(logit.coef_)


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[22]:


array = logit.predict([[ 3.,  0.,  50., 2.]]) #if and else to make a prediction about a passenger for their survival rate
if array[0]==0:
    print("Sorry, you would've died")
else:
    print("Congratulations, you would've survived")


# In[ ]:




