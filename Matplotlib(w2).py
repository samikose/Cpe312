#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

array = np.arange(0,5,0.2)
y = np.sin(array)

plt.plot(array,y)

plt.title("Sinüs Grafiği")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")

plt.show()


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

x = np.random.random(20)
y = np.random.random(20)
colors = np.random.random(20)
sizes = np.random.randint(20,120,20)

plt.scatter(x,y,c=colors,s=sizes)

plt.title("Point Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

data = np.random.random(2000)

plt.hist(data,bins=30)

plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")

plt.show()


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

data1 = ["Bursa","İstanbul","Konya","Karabük","Eskişehir"]
data2 = np.random.randint(1,20,5)


plt.bar(data1,data2)


plt.title("Column Chart")
plt.xlabel("Categories")
plt.ylabel("Values")

plt.show()


# In[12]:


import matplotlib.pyplot as plt


sizes = [16,6,9,12,25,64,24]


plt.pie(sizes)


plt.title("Pie Chart")


plt.show()


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


x = np.arange(-5, 5, 0.50)
y = np.arange(-5, 5, 0.50)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(x, y, z)

ax.set_title("3D Graphics")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# Show
plt.show()


# In[ ]:




