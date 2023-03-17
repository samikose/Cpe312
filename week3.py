#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[4]:


titanic_train = pd.read_csv("C:/Users/samik/OneDrive/Masaüstü/3.sınıf/Machine Learning/train.csv")


# In[8]:


titanic_train.shape


# In[7]:


titanic_train.head()


# In[9]:


del(titanic_train["PassengerId"])


# In[10]:


titanic_train.head()


# In[12]:


new_Pclass = pd.Categorical(titanic_train["Pclass"],ordered=True)
new_Pclass = new_Pclass.rename_categories(["class1","class2","class3"])
titanic_train["Pclass"] = new_Pclass


# In[13]:


new_Pclass.describe()


# In[14]:


titanic_train["Age"].describe()


# In[15]:


missing = np.where(titanic_train["Age"].isnull() == True, )
missing


# In[16]:


len(missing[0])


# In[20]:


titanic_train.hist(column="Age",figsize=(9,6),bins=20)


# In[23]:


new_age_var = np.where(titanic_train["Age"].isnull() == True,28,titanic_train["Age"])
titanic_train["Age"] = new_age_var
titanic_train["Age"].describe()


# In[24]:


titanic_train.hist(column="Age",figsize=(9,6),bins=20)


# # SEABORN

# In[25]:


import seaborn as sns 
import matplotlib.pyplot as plt

data = sns.load_dataset("flights")
data=data.pivot("month","year","passengers")

sns.heatmap(data=data,annot=True,fmt="d",cmap="YlGnBu")

plt.show()


# In[26]:


data.info


# In[27]:


data = sns.load_dataset("flights")

sns.lineplot(data=data,x="year",y="passengers")

plt.show()


# In[28]:


data = sns.load_dataset("flights") 
sns.relplot(x="passengers", y="month", data=data);

plt.show()


# In[29]:


data = sns.load_dataset("iris") 
sns.swarmplot(x="species",y="petal_length",data=data)
plt.show()


# In[30]:


data = sns.load_dataset("tips")
sns.set(style="ticks", color_codes=True)
sns.catplot(x="day", y="total_bill", kind="violin", data=data)
plt.show()


# In[31]:


data = sns.load_dataset("titanic")
sns.countplot(data=data,x="class")
plt.show()


# In[32]:


data = sns.load_dataset("tips")
sns.scatterplot(data=data,x="total_bill",y="tip")
plt.show()


# In[33]:


data = sns.load_dataset("titanic")
a=sns.FacetGrid(data,col="sex")
a.map(sns.histplot,"age")
plt.show()


# In[34]:


data = sns.load_dataset("iris")
g=sns.clustermap(data.drop("species",axis=1),cmap="coolwarm",standard_scale=1)
plt.show()


# In[35]:


sns.set(style="white", color_codes=True)
a = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=a);
sns.despine(offset=10, trim=True);
plt.show()


# # PLOTLY

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go


# In[ ]:





# In[ ]:





# In[67]:


data = pd.read_csv("C:/Users/samik/OneDrive/Masaüstü/3.sınıf/Machine Learning/timesData.csv")


# In[68]:


data.head()


# In[69]:


df2013=data[data.year==2015].iloc[:3,:]
df2013


# In[56]:


trace1 = go.Bar(
x = df2013.university_name, 
y = df2013.citations,  
name = "citations", 
marker = dict(color='rgba(255, 174, 255, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),text = df2013.country)


# In[57]:


trace2 = go.Bar(
x = df2013.university_name, 
y = df2013.teaching,  
name = "teaching", 
marker = dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),text = df2013.country)


# In[58]:


data = [trace1, trace2]


# In[60]:


layout = go.Layout(barmode="group")


# In[61]:


fig = go.Figure(data = data, layout = layout)


# In[62]:


iplot(fig)


# In[70]:


df2014 = data[data.year == 2014].iloc[:100,:]
df2015 = data[data.year == 2015].iloc[:100,:]
df2016 = data[data.year == 2016].iloc[:100,:]


# In[72]:


trace1 =go.Scatter(
x = df2014.world_rank,
y = df2014.citations,
mode = "markers",
name = "2014",
marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
text= df2014.university_name)


# In[73]:


trace2 =go.Scatter(
x = df2015.world_rank,
y = df2015.citations,
mode = "markers",
name = "2015",
marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
text= df2015.university_name)


# In[74]:


trace3 =go.Scatter(
x = df2016.world_rank,
y = df2016.citations,
mode = "markers",
name = "2016",
marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
text= df2016.university_name)


# In[75]:


data = [trace1, trace2, trace3]


# In[76]:


layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False))


# In[77]:


fig = dict(data = data, layout = layout)


# In[78]:


iplot(fig)


# In[81]:


data = pd.read_csv("C:/Users/samik/OneDrive/Masaüstü/3.sınıf/Machine Learning/timesData.csv")


# In[82]:


dataframe = data[data.year == 2015]


# In[83]:


trace1 = go.Scatter3d(
x=dataframe.world_rank,  
y=dataframe.research,  
z=dataframe.citations,  
mode='markers',          
marker=dict(size=dataframe.teaching,color='rgb(255,0,0)',))


# In[84]:


data=[trace1]


# In[85]:


layout=go.Layout(margin=dict(l=0,r=0,b=0,t=0))


# In[86]:


fig = go.Figure(data=data, layout=layout)


# In[87]:


iplot(fig)


# In[88]:


data = pd.read_csv("C:/Users/samik/OneDrive/Masaüstü/3.sınıf/Machine Learning/timesData.csv")


# In[89]:


x2015 = data[data.year == 2015]


# In[90]:


international = [float(each) for each in x2015.international]


# In[91]:


trace0 = go.Box(
y=x2015.international,
name = 'international score of universities in 2015',
marker = dict(color = 'rgb(12, 12, 140)',))
trace1 = go.Box(
y=x2015.research,
name = 'research of universities in 2015',
marker = dict(color = 'rgb(12, 128, 128)',))


# In[92]:


data=[trace0,trace1]


# In[93]:


iplot(data)


# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff


# In[95]:


data = pd.read_csv("C:/Users/samik/OneDrive/Masaüstü/3.sınıf/Machine Learning/timesData.csv")


# In[96]:


dataframe = data[data.year == 2015]


# In[97]:


data2015 = dataframe.loc[:,["research","citations","teaching"]]


# In[98]:


data2015["index"] = np.arange(1,len(data2015)+1)


# In[99]:


data2015.head()


# In[103]:


fig = ff.create_scatterplotmatrix(
data2015,
diag='box',  
index='index', 
colormap='Portland',
colormap_type='cat',
height=700, width=700) 


# In[105]:


iplot(fig)


# In[106]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification as make_ml_clf

COLORS = np.array(
    [
        "!",
        "#FF3333",  # red
        "#0198E1",  # blue
        "#BF5FFF",  # purple
        "#FCD116",  # yellow
        "#FF7216",  # orange
        "#4DBD33",  # green
        "#87421F",  # brown
    ]
)

# Use same random seed for multiple calls to make_multilabel_classification to
# ensure same distributions
RANDOM_SEED = np.random.randint(2**10)


def plot_2d(ax, n_labels=1, n_classes=3, length=50):
    X, Y, p_c, p_w_c = make_ml_clf(
        n_samples=150,
        n_features=2,
        n_classes=n_classes,
        n_labels=n_labels,
        length=length,
        allow_unlabeled=False,
        return_distributions=True,
        random_state=RANDOM_SEED,
    )

    ax.scatter(
        X[:, 0], X[:, 1], color=COLORS.take((Y * [1, 2, 4]).sum(axis=1)), marker="."
    )
    ax.scatter(
        p_w_c[0] * length,
        p_w_c[1] * length,
        marker="*",
        linewidth=0.5,
        edgecolor="black",
        s=20 + 1500 * p_c**2,
        color=COLORS.take([1, 2, 4]),
    )
    ax.set_xlabel("Feature 0 count")
    return p_c, p_w_c


_, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
plt.subplots_adjust(bottom=0.15)

p_c, p_w_c = plot_2d(ax1, n_labels=1)
ax1.set_title("n_labels=1, length=50")
ax1.set_ylabel("Feature 1 count")

plot_2d(ax2, n_labels=3)
ax2.set_title("n_labels=3, length=50")
ax2.set_xlim(left=0, auto=True)
ax2.set_ylim(bottom=0, auto=True)

plt.show()

print("The data was generated from (random_state=%d):" % RANDOM_SEED)
print("Class", "P(C)", "P(w0|C)", "P(w1|C)", sep="\t")
for k, p, p_w in zip(["red", "blue", "yellow"], p_c, p_w_c.T):
    print("%s\t%0.2f\t%0.2f\t%0.2f" % (k, p, p_w[0], p_w[1]))


# In[107]:


import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)

plt.subplot(321)
plt.title("One informative feature, one cluster per class", fontsize="small")
X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1
)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

plt.subplot(322)
plt.title("Two informative features, one cluster per class", fontsize="small")
X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

plt.subplot(323)
plt.title("Two informative features, two clusters per class", fontsize="small")
X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2)
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(324)
plt.title("Multi-class, two informative features, one cluster", fontsize="small")
X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3
)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

plt.subplot(325)
plt.title("Three blobs", fontsize="small")
X1, Y1 = make_blobs(n_features=2, centers=3)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

plt.subplot(326)
plt.title("Gaussian divided into three quantiles", fontsize="small")
X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

plt.show()


# In[108]:


from sklearn import datasets

import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()


# In[109]:


import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()


# In[ ]:




