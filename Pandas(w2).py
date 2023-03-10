#!/usr/bin/env python
# coding: utf-8

# In[2]:


t = (1,2,3,4,5,6,7)
t[0:3]


# In[3]:


import pandas as pd

dictionary = {"name":["sami","veli","caner","seval","ömer","can"],
             "age":[12,24,59,58,None,14],
             "note":[122,466,178,7654,None,189]}

dataframe1 = pd.DataFrame(dictionary) 
print(dataframe1)


# In[ ]:


df = pd.read_csv("C:/Users/Sami/Desktop/data.csv")


print(df)


# In[5]:


head = dataframe1.head()
print(head)


# In[6]:


tail = dataframe1.tail()
print(tail)


# In[7]:


print(dataframe1.columns)


# In[8]:


print(dataframe1.info())


# In[9]:


print(dataframe1.dtypes)


# In[10]:


print(dataframe1.describe()) 


# In[11]:


print(dataframe1["name"]) 
print(dataframe1.loc[:, "age"])                     
dataframe1["yeni_future"] = [1,2,3,4,5,6]
print(dataframe1.loc[:3,"age"]) 
print(dataframe1.loc[:3, "name":"note"])                
print(dataframe1.loc[::-1])


# In[12]:


print(dataframe1.yeni_future) 


# In[14]:


print(dataframe1.loc[:, "name"])


# In[17]:


print(dataframe1.loc[:2,"age"]) 


# In[16]:


print(dataframe1.loc[:2, "name":"age"])


# In[19]:


print(dataframe1.loc[:1, ["name","note"]])


# In[20]:


print(dataframe1.loc[::-1])


# In[22]:


print(dataframe1.loc[:,:"note"])


# In[23]:


print(dataframe1.iloc[:,[3]])


# In[24]:


filtre1 = dataframe1.age>5                           
dataframe1["bool"]= filtre1
print(dataframe1.loc[:,["age","bool"]])


# In[25]:


type(filtre1)


# In[26]:


filtrelenmis_data= dataframe1[filtre1]
print(filtrelenmis_data)


# In[28]:


filtre2 = dataframe1.note>130
filtrelenmis_data2 = dataframe1[filtre2&filtre1]
print(filtrelenmis_data2)


# In[29]:


dataframe1[dataframe1.age>10]


# In[32]:


ortalama = dataframe1.note.mean()
print(ortalama)


# In[31]:


dataframe1.dropna(inplace=True)
dataframe1


# In[33]:


print(dataframe1.note.mean()) 
dataframe1["ortalama"]= ["ortalamanın altında" if dataframe1.note.mean()>each else "ortalamanın üstünde" for each in dataframe1.note]
dataframe1


# In[34]:


dataframe1.columns = [each.upper() for each in dataframe1.columns]
dataframe1.columns


# In[35]:


dataframe1["yeni2_future"]=[1,1,1,1,1]
dataframe1.columns = [each.split('_')[0]+" "+each.split('_')[1] if len(each.split('_'))>1 else each for each in dataframe1.columns]
dataframe1


# In[36]:


dataframe1.columns = [ each.split(" ")[0]+"_"+each.split(" ")[1] if len(each.split(" "))>1 else each for each in dataframe1.columns]
dataframe1


# In[37]:


dataframe1.drop(["yeni2_future","YENI_FUTURE"],axis=1,inplace=True)
dataframe1


# In[38]:


data1 = dataframe1.head()     
data2 = dataframe1.tail()
data_concat = pd.concat([data1,data2],axis=0)
data_concat


# In[39]:


data_contact2 = pd.concat([data1,data2],axis=0)
data_contact2


# In[40]:


dataframe1["buyuk_yas"] = [each*2 for each in dataframe1.AGE]
dataframe1


# In[41]:


def mlt(yas):
    return yas*2
dataframe1["apply_metodu"] = dataframe1.AGE.apply(mlt)
dataframe1


# In[ ]:




