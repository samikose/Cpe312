#!/usr/bin/env python
# coding: utf-8

# In[2]:


num2=23
gün = 'carsamba'

floatnumber = 1.03


# In[3]:


list=[0,2,3,4,5,6,7,8]
type(list)


# In[6]:


list_str = ["pazar","pazartesi","sali","carsamba"]
type(list_str)


# In[7]:


list[4]
list_str[-2]


# In[8]:


list[4]


# In[9]:


list[4:]


# In[10]:


list_str[::2]


# In[11]:


dir(list)


# In[12]:


list.append(25)


# In[13]:


list


# In[14]:


list.append(-25)
list


# In[15]:


list.sort()
list


# In[17]:


list.reverse()
list


# In[19]:


list.remove(-25)
list.sort()
list


# In[20]:


for i in range(1,10):
    print(i)


# In[22]:


for i in "Bursa":
    print(i)


# In[23]:


sum(list)


# In[24]:


min(list)


# In[25]:


minimum = list[0]
for i in list:
    if(i<minimum):
        minimum = i
print(minimum)


# In[26]:


i=0
while (i<9):
    
    print(i)
    
    i+=1


# In[27]:


def dikdortgen_c(x,y):
    return 2*(x+y);
def dikdortgen_a(x,y):
    return x*y


# In[28]:


print(dikdortgen_c(5,6),dikdortgen_a(5,6))


# In[29]:


karesi = lambda x: x*x 
print(karesi(10))


# In[30]:


dictionary = {"bir":1,"iki":2,"üç":3,"dört":4,"beş":5,"altı":6}
type(dictionary)


# In[31]:


print(dictionary["altı"]);


# In[32]:


dictionary.keys()


# In[33]:


dictionary.values()


# In[34]:


def deneme():
    
    dictionary = {"bir":1,"iki":2,"üç":3,"dört":4,"beş":5,"altı":6}
    
    return dictionary

dic = deneme() 
dic["bir"]


# In[35]:


dictionary = {"bir":1,"iki":2,"üç":3,"dört":4,"beş":5,"altı":6}

keys = dictionary.keys() 

if "iki" in keys:
    
    print("evet")

else:
    print("hayır")


# In[ ]:




