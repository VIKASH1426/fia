# stock_forecast.py

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

def stock_forecast_app():
    st.title('Stock Forecast App')

    # Allow the user to input any stock symbol
    user_input_stock = st.text_input('Enter the stock symbol (e.g. AAPL, TSLA, GOOGL):', 'AAPL')

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Load data based on user input
    data_load_state = st.text('Loading data...')
    data = load_data(user_input_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Recommendation Logic based on the forecast
    def get_recommendation(forecast, current_price):
        predicted_prices = forecast['yhat'].tail(period)

        first_predicted = predicted_prices.iloc[0]
        last_predicted = predicted_prices.iloc[-1]

        price_change = (last_predicted - current_price) / current_price * 100

        if price_change > 5:
            recommendation = "Buy"
            reason = f"The stock price is predicted to increase by {price_change:.2f}% over the next {n_years} year(s). This suggests a good opportunity to buy."
        elif price_change < -5:
            recommendation = "Sell"
            reason = f"The stock price is predicted to decrease by {price_change:.2f}% over the next {n_years} year(s). You may want to sell to avoid potential losses."
        else:
            recommendation = "Hold"
            reason = f"The stock price is predicted to change by {price_change:.2f}% over the next {n_years} year(s), indicating stability. You might want to hold your position."

        return recommendation, reason

    # Get the last closing price
    current_price = data['Close'].iloc[-1]

    # Generate recommendation
    recommendation, reason = get_recommendation(forecast, current_price)

    # Display recommendation
    st.subheader('Recommendation')
    st.write(f"**Recommendation:** {recommendation}")
    st.write(f"**Reason:** {reason}")
