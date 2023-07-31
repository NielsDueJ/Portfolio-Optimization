#!/usr/bin/env python
# coding: utf-8

# ## Simulated Portfolio Optimization using Max Sharpe and Min Vol

# In[1]:


#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 4)


# In[2]:


#Plot configuration
plt.style.use('grayscale')
np.random.seed(123)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[3]:


#Importing data on daily closing prices
data = pd.read_excel("/Users/nielsduejensen/Desktop/Databook_Returns.xlsx", 'Lærerenes Combined Prices',
                    header=0, index_col=0)


# In[4]:


data.head(3)


# In[5]:


#Convert daily closing prices to returns
returns = data.pct_change()


# In[6]:


#The following equations generates randomly weighted portfolios


# In[7]:


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


# In[8]:


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(996)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
        


# In[9]:


#Input parameters
returns = data.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 10000
risk_free_rate = 0.0


# In[10]:


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=data.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=data.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    
    fig = plt.figure(figsize=(10,7))
    gs = GridSpec(4, 4)
    
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    #plt.figure(figsize=(13, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.scatter(sdp,rp,marker='.',color='g',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='.',color='steelblue',s=500, label='Minimum volatility')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8,loc=0)
    
    ax_hist_y = fig.add_subplot(gs[0,0:3])
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    plt.title('Lærerenes Combined Portfolio')
    plt.hist(results[0,:],bins=100,color = "lightseagreen")
    ax_hist_x = fig.add_subplot(gs[1:4, 3])
    plt.hist(results[1,:],orientation="horizontal",bins=100,color = "lightseagreen")
    ax_hist_x.tick_params(top=False,
               bottom=True,
               left=False,
               right=False,
               labelleft=False,
               labelbottom=True)

    plt.savefig('Documents/newplots/Lærerenes_Combined.pdf', bbox_inches='tight')


# In[11]:


display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)


# In[ ]:




