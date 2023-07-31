#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns


# In[2]:


prices = pd.read_excel('/Users/nielsduejensen/Desktop/Data_Book_Rets.xlsx', 'Danica Exclusion Prices',header=0, index_col=0)


# In[3]:


prices.head(3)


# In[4]:


returns = prices.pct_change().dropna()


# In[5]:


prices.iloc[0]


# In[6]:


meanreturns = returns.mean()
covmatrix = returns.cov()


# In[7]:


weights = np.random.random(len(meanreturns))
weights /= np.sum(weights)


# ## Monte Carlo Simulation

# In[8]:


mc_sims = 1000
T = 252


# In[9]:


meanM = np.full(shape=(T, len(weights)), fill_value=meanreturns)
meanM = meanM.T


# In[10]:


portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)


# In[11]:


initialportfolio = 100


# In[12]:


for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = covmatrix
    returns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, returns.T)+1)*initialportfolio


# In[13]:


plt.figure(figsize=(9,5.5))
sns.set_theme(style="ticks")
sns.set_context("talk")
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Danica Exclusion Portfolio')
plt.axhline(initialportfolio,c='k')
plt.plot(np.mean(portfolio_sims))
plt.show()
plt.savefig('Documents/plots/MC2_Danica_Exclusion.pdf', bbox_inches='tight')


# In[14]:


def mcVaR(returns, alpha=5):
    """INput: pandas series of reutnrs
    Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")


# In[15]:


def mcCVaR(returns, alpha=5):
    """Input: pandas series of returns
    Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")


# In[16]:


portResults = pd.Series(portfolio_sims[-1,:])


# In[17]:


VaR_I = mcVaR(portResults, alpha=1)


# In[18]:


VaR = initialportfolio - mcVaR(portResults, alpha=1)


# In[19]:


CVaR = initialportfolio - mcCVaR(portResults, alpha=1)


# In[20]:


VaR_I.round(2)


# In[21]:


VaR.round(2)


# In[22]:


CVaR


# In[29]:


portResults.mean()


# In[ ]:




