#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt 
from matplotlib import rc
import pylab as plb
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('/Users/kf/Downloads/case_4_accuracy.csv')


# In[3]:


standErr = np.std(data, ddof=1, axis=1) / np.sqrt(np.size(data, axis =1))


# In[4]:


data['mean'] = data.mean(axis=1)


# In[15]:


plt.figure(figsize=(7,8)) 
plt.errorbar( data['Unnamed: 0'], data['mean'], yerr=standErr, fmt='o', color='Black', elinewidth=1,capthick=1,errorevery=1, alpha=1, ms=1, capsize = 3)
plt.bar(data['Unnamed: 0'], data['mean'],tick_label = data['Unnamed: 0'], color = ['mediumpurple', 'cyan', 'b','r'],edgecolor='black')##Bar plot
plt.yticks(rotation = 90)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Classifier Type", size = 15)
plt.ylabel("Accuracy", size = 15)
ax = plt.gca()
yticks = ax.yaxis.get_major_ticks() 
yticks[0].label1.set_visible(False)
plt.show()


# In[ ]:




