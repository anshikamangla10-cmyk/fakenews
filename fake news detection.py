#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[3]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('true.csv')


# In[4]:


data_fake.head()


# In[5]:


data_true.head()


# In[6]:


import pandas as pd
data_fake = pd.read_csv('C:/Users/hp/Documents/Projects/Fake news detection/fake.csv')
data_true = pd.read_csv('C:/Users/hp/Documents/Projects/Fake news detection/True.csv')
data_fake["class"] = 0
data_true['class'] = 1


# In[7]:


data_fake.shape, data_true.shape


# In[8]:


data_fake_manual_testing = data_fake.tail(10)
data_fake = data_fake.iloc[:-10]
    
data_true_manual_testing = data_true.tail(10)
data_true = data_true.iloc[:-10]


# In[9]:


print("Fake data shape after removal:", data_fake.shape)
print("True data shape after removal:", data_true.shape)


# In[10]:


data_fake_manual_testing = data_fake.tail(10).copy()
data_fake_manual_testing['class'] = 0
data_true_manual_testing = data_true.tail(10).copy()
data_true_manual_testing['class'] = 1


# In[11]:


data_fake_manual_testing.head(10)


# In[12]:


data_true_manual_testing.head(10)


# In[13]:


data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge.head(10)


# In[14]:


import pandas as pd
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')
data_fake['class'] = 0
data_true['class'] = 1
data_merge = pd.concat([data_fake, data_true], axis=0)
print(data_merge.columns)


# In[15]:


data = data_merge.drop(['title','subject', 'date'],axis=1)


# In[16]:


data.isnull().sum()


# In[17]:


data = data.sample(frac = 1)


# In[18]:


data.head()


# In[19]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)


# In[20]:


data.columns


# In[21]:


data.head(10)


# In[22]:


def wordopt(text):
    text = re.sub(r'\[.*?\]', '', text)                              # remove content in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)               # remove URLs
    text = re.sub(r'<.*?>+', '', text)                               # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)  # remove punctuation
    text = re.sub(r'\n', '', text)                                   # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)                             # remove words with digits
    return text


# In[23]:


data['text'] = data['text'].apply(wordopt)


# In[24]:


x = data['text']
y = data['class']


# In[25]:


x_train ,x_test ,y_train ,y_test = train_test_split(x,y, test_size= 0.25)


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[27]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[28]:


pred_lr = LR.predict(xv_test)


# In[29]:


LR.score(xv_test, y_test)


# In[30]:


print(classification_report(y_test, pred_lr))


# In[31]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[32]:


pred_DT = DT.predict(xv_test)


# In[33]:


DT.score(xv_test, y_test)


# In[34]:


print(classification_report(y_test, pred_DT))


# In[35]:


from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(n_estimators=20, random_state = 0)
GB.fit(xv_train, y_train)


# In[36]:


predict_gb = GB.predict(xv_test)


# In[37]:


GB.score(xv_test, y_test)


# In[38]:


print(classification_report(y_test, predict_gb))


# In[39]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)


# In[40]:


predict_rf = RF.predict(xv_test)


# In[41]:


RF.score(xv_test, y_test)


# In[42]:


print(classification_report(y_test, predict_rf))


# In[43]:


def output_label(n):
    if n == 0:
        return "Fake news"
    elif n == 1:
        return "Not a Fake news"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GB.predict(new_xv_test)
    pred_RFC = RF.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \n\nDT Prediction: {} \n\nGBC Prediction: {} \n\nRFC Prediction: {}".format(output_label(pred_LR[0]),
                                                                                                                    output_label(pred_GB[0]),
                                                                                                                    output_label(pred_RF[0])))


# In[ ]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:




