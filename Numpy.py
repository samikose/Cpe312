#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


array = np.array([1,2,3,4,5])
print(array)


# In[4]:


array2= np.array([1,2,3,4,5,6,7,8,9,10])
print(array2.shape)


# In[6]:


a = array2.reshape(2,5) 
print(a)


# In[7]:


print("shape: ",a.shape)

print("dimension: ",a.ndim)

print("data type: ",a.dtype.name)

print("size: ",a.size)

print("type",type(a))


# In[8]:


array3= np.array([[1,2,3,4],[5,6,7,8]])


# In[9]:


print(array3)


# In[10]:


print(array3.shape)


# In[13]:


zeros= np.zeros((4,4))
print(zeros)


# In[12]:


zeros[0,0]=5
print(zeros)


# In[17]:


np.ones((3,3))


# In[19]:


np.empty((2,4))


# In[20]:


np.arange(10,40,2)


# In[21]:


a= np.linspace(0,10,10)
print(a)


# In[22]:


a= np.array([1,2,3])
b= np.array([2,4,9])
print(a+b) 
print(a-b) 
print(a**2)


# In[23]:


a= np.array([1,2,3,4])


# In[24]:


liste = [1,2,3,4]
array = np.array(liste)
print(array)
print(liste)


# In[25]:


liste2 = list(array)
print(liste2)


# In[26]:


a = np.array([1,2,3,4,5,6])
print(a)


# In[27]:


b=a

c=a

b[0]=2

print(a,b,c)


# In[28]:


d = a.copy()


# In[29]:


a= np.array([1,2,3,4])
print(a)


# In[30]:


print(a[2])


# In[31]:


print(a[0:3])


# In[32]:


reverse_array = a[::-1]
print(reverse_array)


# In[34]:


b = np.array([[1,2,3,4,5,6],[6,7,8,9,10,11]]) 
print(b)


# In[39]:


print(b[1,1])

print(b[:,2]) 

print(b[1,:]) 

print(b[1,1:6]) 

print(b[-1,:]) 

print(b[:,-1])


# In[40]:


array = np.array([[2,4,9],[16,25,36],[49,64,81]]) #3*3'lük array oluşturuldu.
print(array)


# In[42]:


a = array.flatten() 
print(a)
a = array.ravel()
print(a)


# In[43]:


array2 = a.reshape(3,3)
print(array2)


# In[44]:


array_transpose = array2.T
print(array_transpose)


# In[45]:


array3 = np.array([[1,2],[3,4],[5,6],[7,8]])
print(array3)


# In[47]:


array3.resize((2,4))
print(array3)


# In[ ]:




