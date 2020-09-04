# Time_Series_Models
This repo contains datasets from JetRail Problem (AV competition) and model built upon those datasets for time series forcasting

I have used various techniques used for time series forecasting like ARIMA, SARIMAX(through PMD_AutoArima), FBProphet and RNN-LSTM and measured their performance through RMSE and MAPE metric.
There are separate files for detailed EDA for finding out stationarity, seasonality, trend and residuals of the data given.
To find out the parameters there are various plots like ACF & PACF. Seasonal_Decompose and ADFuller Test are also performed to break the components of time series.
Best Model for hourly forecasting was LSTM with an R2 score of around 94-95%.
Best model for daily forecasting was SARIMA.
