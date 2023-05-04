#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 




stock=pd.read_csv("C:\\Users\\LENOVO\\Downloads\\1729258-1613615-Stock_Price_data_set_(1).csv")
stock.head()


# In[2]:


stock.info()


# In[18]:


stock['Date'] = pd.to_datetime(stock['Date'])


# # This output contains ,stock  price between dates,Total days

# In[19]:


print(f'Dataframe contains stock prices between {stock.Date.min()} {stock.Date.max()}')
print(f'Total days={(stock.Date.max() - stock.Date.min()).days} days')


# THIS GIVES INFORMATION REGARDING DATA

# In[ ]:


stock.describe()


# In[6]:


stock[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# In[ ]:


#setting layout for our plot 


# In[4]:


layout = go.Layout(
    title='Stock Prices ',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

stock_data = [{'x':stock['Date'], 'y':stock['Close']}]
plot = go.Figure(data=stock_data, layout=layout)


# In[5]:


iplot(plot)


# In[8]:


# Building the regression model
from sklearn.model_selection import train_test_split

#For preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#For model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[9]:


#Split the data into train and test sets
X = np.array(stock.index).reshape(-1,1)
Y = stock['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[2]:


# Feature scaling
scaler = StandardScaler().fit(X_train)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


#Creating a linear model
lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[4]:


#Plot actual and predicted values for train dataset
trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
stock_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=stock_data, layout=layout)


# In[14]:


iplot(plot2)


# In[15]:


#Calculate scores for model evaluation
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)


# In[3]:


print("Goodbye!")


# In[ ]:




