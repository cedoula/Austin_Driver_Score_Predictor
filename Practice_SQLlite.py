#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3

conn = sqlite3.connect('practice.db')

print ("Opened database successfully")


# In[7]:


conn.execute('''CREATE TABLE MOCK_DATA
         (IDX INT PRIMARY KEY     NOT NULL,
         Day_of_Week           TEXT    NOT NULL,
         Latitude            INT     NOT NULL,
         Longitude        INT          NOT NULL,
         Person_Age         INT        NOT NULL,
         Vehicle_Make       TEXT    NOT NULL);''')
print ("Table Created Successfully")


# In[23]:


file = "Resources/mock_data_for_sql.csv"


# In[26]:


import pandas as pd
data = pd.read_csv(file, index_col='IDX')


# In[16]:


# data.to_sql("MOCK_DATA", conn, if_exists = 'append', index='IDX')


# In[27]:


data.head()


# In[28]:


data.to_sql("MOCK_DATA", conn, if_exists = 'append', index='IDX')


# In[30]:


conn.execute("SELECT * FROM MOCK_DATA").fetchall()


# In[31]:


dataframe = conn.execute("SELECT * FROM MOCK_DATA").fetchall()


# In[35]:


cols=['IDX','Day_of_Week', 'Latitude','Longitude','Person_Age','Vehicle_Make']


# In[36]:


df_new = pd.DataFrame(dataframe, columns=cols)


# In[37]:


df_new.head()


# In[ ]:




