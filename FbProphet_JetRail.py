#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv(r'C:\Users\tyagir01\Desktop\Train_SU63ISt.csv')
df.head()


# In[3]:


df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
df.index = df.Timestamp 
df = df.resample('D').mean()


# In[4]:


from fbprophet import Prophet


# In[5]:


df = df.reset_index()[['Datetime','Count','ID']].rename({'Datetime':'ds','Count':'y','ID':'ID'}, axis='columns')


# In[6]:


df.head()


# In[7]:


model = Prophet(daily_seasonality=True)
model.fit(df)


# In[28]:


#model.params


# In[8]:


future = model.make_future_dataframe(periods=212)
future.tail()


# In[9]:


forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[10]:


plt.figure(figsize=(20,10))
fig = model.plot(forecast)


# In[11]:


comp = model.plot_components(forecast)


# In[12]:


base_forecast = forecast[:762]


# In[13]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(df.y, base_forecast.yhat))
rmse


# In[15]:


## Cross Validation
from fbprophet.diagnostics import cross_validation , performance_metrics
cv_results = cross_validation(model=model, initial = '608 days', horizon = '153 days')
df_p = performance_metrics(cv_results)
df_p


# In[16]:


from fbprophet.plot import plot_cross_validation_metric
fig2 = plot_cross_validation_metric(cv_results, metric='mape')


# In[17]:


fig3 = plot_cross_validation_metric(cv_results, metric='rmse')


# In[18]:


final_predictions = forecast[['ds','yhat']][762:]
final_predictions.index = final_predictions.ds
final_predictions.index.name = 'Datetime'
final_predictions = final_predictions.rename(columns= {'yhat': 'Count'})
final_predictions.drop(columns='ds', inplace=True)
final_predictions


# In[19]:


final_predictions = final_predictions.round(0).astype(int)
final_predictions


# In[20]:


final_predictions.to_csv('Time_Series_JetRail_Predictions.csv')

