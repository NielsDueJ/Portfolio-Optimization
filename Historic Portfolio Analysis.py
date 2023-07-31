#!/usr/bin/env python
# coding: utf-8

# ## Portfolio Analysis on Historic Prices

# In[1]:


#Importing packages
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Reading data on daily closing prices
Daily_closing_prices = pd.read_excel('/Users/nielsduejensen/Desktop/Data_Book_Rets.xlsx', 'Sampension Exclusion Prices',header=0, index_col=0)


# In[3]:


Daily_closing_prices.head(3)


# In[4]:


#Deriving the returns from the price data
ret = Daily_closing_prices.pct_change().dropna()
ret.mean(axis = 1)
#Defining number of assets
no_assets = len(Daily_closing_prices.columns)
no_assets
#Defining weights
weights = [1/no_assets for i in range(no_assets)]
#Assigning weights and returns
ret.mul(weights, axis = "columns")
ret.dot(weights)
#Constructing EW-Portfolio
ret["EWP"] = ret.dot(weights)
pr = pd.DataFrame([ret["EWP"]])
portfolio_returns = pr.T
#Defining returns of EW-Portfolio
portfolio_returns


# In[5]:


portfolio_returns.std()
portfolio_returns.mean()
#Computing volatility
deviations = portfolio_returns - portfolio_returns.mean()
squared_deviations = deviations**2
variance = squared_deviations.mean()
volatility = np.sqrt(variance)


# In[6]:


print(portfolio_returns.std())
print(portfolio_returns.mean())
print(volatility)


# In[7]:


#Plotting daily returns
plt.figure(figsize=(14, 7))
for i in portfolio_returns.columns.values:
    plt.plot(portfolio_returns.index, portfolio_returns[i], lw=2, alpha=0.8)
plt.legend(loc='lower center', fontsize=14)
plt.title("Danica Exclusion Portfolio Correlations")
plt.ylabel('Daily returns')
plt.xlabel("Time")


# In[8]:


portfolio_returns.shape[0]
n_days = portfolio_returns.shape[0]
## Annualized Volatility
annualized_vol = portfolio_returns.std()*np.sqrt(252)
## Annualized Return
n_days = portfolio_returns.shape[0]
return_per_day = (portfolio_returns+1).prod()**(1/n_days) - 1
annualized_return = (portfolio_returns+1).prod()**(252/n_days) - 1
#Sharpe ratio
riskfree_rate = 0.00
excess_return = annualized_return - riskfree_rate
sharpe_ratio = excess_return/annualized_vol
print(annualized_vol)
print(annualized_return)
print(sharpe_ratio)


# ## Extreme Risk Analysis

# In[9]:


wealth_index = 1000*(1+portfolio_returns).cumprod()
wealth_index.head()
wealth_index.plot.line()


# In[10]:


previous_peaks = wealth_index.cummax()
previous_peaks.plot()


# In[11]:


drawdown = (wealth_index - previous_peaks)/previous_peaks
drawdown.plot()


# In[12]:


drawdown.min()


# In[13]:


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


# In[14]:


drawdown(portfolio_returns["EWP"]).head()


# In[16]:


drawdown(portfolio_returns["EWP"])[["Wealth", "Previous Peak"]].head()


# In[17]:


drawdown(portfolio_returns["EWP"])[["Wealth", "Previous Peak"]].plot()


# In[ ]:




