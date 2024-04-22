import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('Stock Trend Predictions Model.keras')

background_image_url = 'https://www.pexels.com/photo/round-silver-and-gold-coins-730564/'

st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: url("{'https://www.pexels.com/photo/round-silver-and-gold-coins-730564/'}") no-repeat center center fixed;
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


st.header('Stock Market Predictor')
stock=st.text_input('Enter Stock Symbol', 'AAPL')
start='2012-01-01'
end='2022-12-31'

data=yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index = True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price Vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price Vs MA50 Vs MA100 ')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price Vs MA100 Vs MA200 ')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_
predict = predict*scale
y=y*scale

st.subheader('Original Price Vs Predicted Price ')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,'r', label='Original Price')
plt.plot(y,'g', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)