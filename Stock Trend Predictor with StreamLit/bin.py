import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yfin
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

yfin.pdr_override()


st.title("Stock Trend Predictor")
input_name=st.text_input('Enter Stock Ticker', 'AAPL')

for_date = int(str(date.today())[:4])-10
start = str(for_date)+str(date.today())[4:]
end = str(date.today())

df = pdr.get_data_yahoo(input_name, start, end)

st.subheader("Data from {0} to {1}:".format(start,end))

df = df.reset_index()
df = df.drop(["Date"], axis=1)

st.write(df.describe())

st.subheader("Stock Closing Price vs Time:")
fig_close = plt.figure(figsize=(12,6))
plt.xlabel("Time")
plt.ylabel("Price")
plt.plot(df.Close, label='Closing Price')
plt.legend(['Closing Price'])
st.pyplot(fig_close)

st.subheader("Stock Closing Price along with Moving Average(100) vs Time:")
ma100 = df.Close.rolling(100).mean()
fig_ma100 = plt.figure(figsize=(12,6))
plt.xlabel("Time")
plt.ylabel("Price")
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100 , 'r')
plt.legend(['Closing Price','Moving Average(100)'])
st.pyplot(fig_ma100)

st.subheader("Stock Closing Price along with Moving Average(100 and 200) vs Time:")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig_ma200 = plt.figure(figsize=(12,6))
plt.xlabel("Time")
plt.ylabel("Price")
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'g')
plt.legend(['Closing Price','Moving Average(100)','Moving Average(200)'])
st.pyplot(fig_ma200)

#splitting data
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70):])

#scaling data in between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
train_arr = scaler.fit_transform(train)

#loading the model
model = load_model("stock_trend_predictor.h5")

#preparing testing data
past_100 = train.tail(100)
final_test= past_100.append(test,ignore_index=True)

final_test_arr = scaler.fit_transform(final_test)

x_test , y_test = [],[]
for i in range(100,final_test_arr.shape[0]):
    x_test.append(final_test_arr[i-100:i])
    y_test.append(final_test_arr[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader("Predicted and Actual Values vs Time:")
fig_pred = plt.figure(figsize=(12,6))
plt.xlabel("Time")
plt.ylabel("Price")
plt.plot(y_test , 'b')
plt.plot(y_pred , 'r')
plt.legend(['Original Values','Predicted Values'])
plt.plot()
st.pyplot(fig_pred)