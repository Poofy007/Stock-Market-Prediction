import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

def plot_test_results(pred, y_test):
    fig = plt.figure()
    plt.title('Predicted Trends v. Real Trends on Test Dataset')
    plt.plot(pred, color='green', label='Predictions')
    plt.plot(y_test, color='red', label='Truth')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    return fig

def plot_future(pred):
    fig = plt.figure()
    plt.title('Predicted Trend For Next 3 Days')
    plt.plot(pred, color='blue')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    return fig

st.title('Basic RNN Stock Price Trend Predictor')
st.write('Description [placeholder]')
st.write('Instructions: Upload a csv file. Make sure there is a column with the dates labeled as \'Dates\', and a label with the closing prices \'Close\'.')

uploaded_file = st.file_uploader("Choose a CSV file", type = ['csv', 'txt', 'xlsx'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'High' not in df.columns:
        st.error('Dataframe is missing column labeled \'Close\'')
    elif 'Date' not in df.columns:
        st.error('Dataframe is missing column labeled \'Date\'')
    else:
        df['datetimedate'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='datetimedate')
        df['year'] = df['datetimedate'].dt.year

        # get the last 1000 days
        df = df[-1000:]

        df = df.drop(columns=['Open','High','Low','Volume','OpenInt'])

        # split train and test data
        close = np.array(df.Close).reshape(-1,1)
        threshold = int(len(close)*0.9)
        train_close = close[0:threshold]
        test_close = close[threshold:]

        # normalize
        scaler = MinMaxScaler()
        scaler.fit(train_close)
        train_scaled = scaler.transform(train_close)

        # sliding window
        x_train = []
        y_train = []
        window_size = 30

        for i in range(len(train_scaled) - window_size):
            x_train.append(train_scaled[i:i + window_size])
            y_train.append(train_scaled[window_size + i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # model
        regressor = Sequential()
        regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True, input_shape = (window_size,1)))
        regressor.add(Dropout(0.2))
        regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = False))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))

        # train
        regressor.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3), loss = "mean_squared_error")
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
        epochs = 120
        batch_size = 32
        with st.spinner('Training Model (This will take a while)...'):
            regressor.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, callbacks=[callback])

        # test data set up
        test_scaled = scaler.transform(test_close)

        x_test = []
        y_test = []
        window_size = 30

        for i in range(len(test_scaled) - window_size):
            x_test.append(test_scaled[i:i + window_size])
            y_test.append(test_scaled[window_size + i])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # estimation of trend for next 3 days (from the last day in the dataset)
        x = test_scaled[-30:]
        preds = []

        for i in range(3):
            p = regressor.predict(x[np.newaxis, ...])
            preds.append(p.item())
            x = x[1:]
            x = np.append(x, p, axis=0)

        pred = scaler.inverse_transform(np.array(preds)[..., np.newaxis])

        st.pyplot(plot_future(pred))
        st.caption('Trend prediction for the next 3 days - after the last day in the dataset (3-day lookahead).')

        # predict on test dataset
        pred = regressor.predict(x_test)
        pred = scaler.inverse_transform(pred)
        y_test = scaler.inverse_transform(y_test)

        st.pyplot(plot_test_results(pred, y_test))
        st.caption('Performed with a 1-day lookahead using a RNN')