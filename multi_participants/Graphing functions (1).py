#!/usr/bin/env python
# coding: utf-8

# In[26]:


from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np


# In[27]:


data = pd.read_csv('/Users/kf/Downloads/case_4_accuracy.csv')


# In[28]:


standErr = np.std(data, ddof=1, axis=1) / np.sqrt(np.size(data, axis =1))


# In[29]:


data['mean'] = data.mean(axis=1)


# In[30]:


plt.figure(figsize=(7,8)) 
plt.errorbar( data['Unnamed: 0'], data['mean'], yerr=standErr, fmt='o', color='Black', elinewidth=2,capthick=2,errorevery=1, alpha=1, ms=2, capsize = 3)
plt.bar(data['Unnamed: 0'], data['mean'],tick_label = data['Unnamed: 0'], color = 'mediumslateblue')##Bar plot


# In[ ]:




