#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns


# In[68]:


doc = pd.read_csv("/Users/abhinavpathak/Downloads/" + 'DS_interview_doc.csv')
paid_unlock = pd.read_csv("/Users/abhinavpathak/Downloads/" + 'DS_interview_paid_unlock.csv')


# In[69]:


print(doc.shape)
print(paid_unlock.shape)
doc.head()


# In[57]:


print(paid_unlock.shape)
paid_unlock.head()


# In[58]:


df = pd.merge(paid_unlock, doc[['New_ID','date']], how = "right", on = 'New_ID')
df['days_diff'] = (pd.to_datetime(df.unlock_date) - pd.to_datetime(df.date))
df['days_diff'] = df['days_diff'].apply(lambda x: x.days)

df['new_unlock_cnt'] = np.where(df['days_diff'] < 365*2, df.unlock_cnt, 0)
df2 = df.groupby('New_ID').new_unlock_cnt.sum().reset_index()
df2['high_quality_flag'] = np.where( df2.new_unlock_cnt >= 3, 1, 0)
print("percentage of quality documents = ", round((df2.high_quality_flag.sum()/df2.shape[0])*100,1), "%")


# In[64]:


for col in doc.select_dtypes(['object']).columns:
    print(col, doc[col].nunique())


# In[60]:


doc = pd.merge(doc, df2, how='left', on = 'New_ID')


# In[14]:


doc.describe()


# In[53]:


# % of missing values in each column
doc.isnull().sum()/doc.shape[0]*100


# ### Missing values imputation

# In[16]:


#remove 'is_mcq' as 95% values are missing and therefore variable cannot be imputed
doc.drop(['is_mcq'], axis = 1, inplace= True)

#replacing missing enrollment with median enrollment
doc['enrollment'].fillna(doc.enrollment.median(), inplace = True)

#replacing missing school country with Mode
doc['school_country'] = np.where(doc.school_country == 'missing', doc.school_name.mode()[0], doc.school_country)

#replacing missing school name with Mode
doc['school_name'] = np.where(doc.school_name == 'missing', doc.school_name.mode()[0], doc.school_name)

# imputing language variable based on Country level modes
doc1 = doc.copy()
frames = []
for i in list(set(doc1['school_country'])):
    df_country = doc1[doc1['school_country']== i]
    try:
        df_country['language'].fillna(df_country['language'].mode()[0],inplace = True)
    except:
        df_country['language'].fillna(doc['language'].mode()[0],inplace = True)
    frames.append(df_country)
    final_df = pd.concat(frames)
    
# Correcting for language codes
lang_map = {'en':'english',
'de':'german',
'es':'spanish',
'fr':'french',
'ja':'japanese',
'sv':'swedish'}

final_df['language'] = np.where(final_df.language.isin(['en','de','es','fr','ja','sv']),final_df['language'].map(lang_map),final_df.language)
doc = final_df


# imputing hqd_scores variable based on tag level mean, as the readability depends on Tags
doc1 = doc.copy()
frames = []
for i in list(set(doc1['tag'])):
    df = doc1[doc1['tag']== i]
    try:
        df['hqd_score'].fillna(df['hqd_score'].mean(),inplace = True)
    except:
        df['hqd_score'].fillna(doc['hqd_score'].mean(),inplace = True)
    frames.append(df)
    final_df = pd.concat(frames)
doc = final_df


# In[17]:


doc[doc.school_type =='missing']


# In[18]:


doc.isnull().sum()


# In[19]:


doc_prop = doc.groupby('year_month').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.lineplot(x = doc_prop.index, y = doc_prop.prop, data = doc_prop)


# In[20]:


plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.boxplot(x = doc.high_quality_flag, y = np.log(doc.byte_size + 1))


# In[21]:


doc_prop = doc.groupby('filetype').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
doc_prop = doc_prop[(doc_prop.prop != 0) & (doc_prop.New_ID >= 10)]
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x = doc_prop.index, y = doc_prop.prop)


# In[22]:


doc_prop = doc.groupby('school_country').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
doc_prop = doc_prop[(doc_prop.prop != 0) & (doc_prop.New_ID >= 10)]
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x = doc_prop.index, y = doc_prop.prop)


# In[23]:


doc_prop = doc.groupby('school_name').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
doc_prop = doc_prop[(doc_prop.prop != 0) & (doc_prop.New_ID >= 50)]
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x = doc_prop.index, y= doc_prop.prop)


# In[24]:


doc_prop = doc.groupby('school_type').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x = doc_prop.index, y = doc_prop.prop)


# In[25]:


plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.boxplot(x = doc.high_quality_flag, y= np.log(doc.enrollment+1))


# In[27]:


doc_prop = doc.groupby('first_subject_name').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
doc_prop = doc_prop[(doc_prop.prop != 0) & (doc_prop.New_ID >= 10)]
plt.figure(figsize = (16,4))
plt.xticks(rotation = 80, color = 'black')
sns.barplot(x = doc_prop.index, y = doc_prop.prop)


# In[28]:


doc_prop = doc.groupby('tag').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x = doc_prop.index, y = doc_prop.prop)


# In[66]:


plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.boxplot(x = doc.high_quality_flag, y = np.log(doc.page_count+1 ))
plt.title("Page Count")


# In[30]:


doc.groupby('high_quality_flag').page_count.mean()


# In[51]:


plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.boxplot(x = doc.high_quality_flag, y = (doc.hqd_score*100))


# In[32]:


doc_prop = doc.groupby('top_keywords').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x = doc_prop.index, y = doc_prop.prop)


# In[34]:


doc_prop = doc.groupby('language').agg({'high_quality_flag':'sum', 
                              'New_ID':'count'})
doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
doc_prop = doc_prop[(doc_prop.prop != 0) & (doc_prop.New_ID >= 5)]
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x =doc_prop.index, y = doc_prop.prop)


# In[44]:


doc_prop = doc.groupby('school_country').agg({'hqd_score':'mean', 
                              'New_ID':'count'})
#doc_prop['prop'] = round(doc_prop.high_quality_flag/doc_prop.New_ID, 2)
doc_prop = doc_prop[(doc_prop.hqd_score != 0) & (doc_prop.New_ID >= 25)]
plt.figure(figsize = (16,4))
plt.xticks(rotation = 60, color = 'black')
sns.barplot(x =doc_prop.index, y = doc_prop.hqd_score)


# In[70]:


# Checking how tags influence page count
doc.groupby('tag')['page_count'].mean()


# In[37]:


from sklearn import tree
from sklearn.preprocessing import OneHotEncoder


# In[38]:


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

x = doc[['byte_size', 'filetype',
       'school_country', 'school_type', 
         'enrollment',
       'first_subject_name', 'tag', 
         'page_count', 'hqd_score',
         #'is_mcq',
       'top_keywords'
         ,'language']]

features_to_encode = ['filetype','school_country','school_type', 'first_subject_name', 'tag',
                     'language']
for feature in features_to_encode:
    res = encode_and_bind(x, feature)
    x = res
    print(feature)
    
x.fillna(0, inplace = True)


# In[39]:


y = doc.high_quality_flag
clf = tree.DecisionTreeClassifier(max_depth = 5)
clf.fit(x,y)

fig = plt.figure(figsize=(20,20))
_ = tree.plot_tree(clf, fontsize = 10, feature_names = x.columns, filled = True)


# In[40]:


importance = clf.feature_importances_
# summarize feature importance
imp = {}
for i,v in enumerate(importance):
    imp[x.columns[i]] = round(v,2)
    
sorted(imp.items(), key=lambda item: item[1], reverse = True)[:10]

