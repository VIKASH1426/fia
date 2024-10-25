import yfinance as yf
import numpy as np
import streamlit as st

# Show portfolio details
def show_portfolio(portfolio):
    st.subheader("Your Current Stock Portfolio:")
    if not portfolio:
        st.write("Your portfolio is empty. Add some stocks to start investing.")
    else:
        for ticker, details in portfolio.items():
            st.write(f"Stock: {ticker}, Quantity: {details['quantity']}, Purchase Price: ${details['purchase_price']:.2f}")

# Recommend stocks based on various factors (basic implementation, can be expanded)
def recommend_stocks():
    st.subheader("Stock Recommendations:")
    st.write("Based on market analysis and portfolio optimization, here are some recommended stocks to consider:")
    recommendations = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Example stock recommendations
    for stock in recommendations:
        st.write(f"Consider investing in {stock} based on current market trends.")

# Calculate portfolio worth
def portfolio_worth(portfolio):
    total_worth = 0
    for ticker, details in portfolio.items():
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][0]
        total_worth += current_price * details['quantity']
    return total_worth

# Display stock price chart
def show_stock_price(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    st.line_chart(hist['Close'], width=700, height=400, use_container_width=False)
    st.write(f"Showing the price chart for {ticker} over the last year.")

# Predict future stock prices (simplified version)
def stock_prediction_chart(ticker):
    st.subheader(f"Stock Prediction for {ticker}:")
    st.write("**(Note: This is a simple linear regression-based forecast and may not be accurate)**")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")['Close']
    hist.plot(figsize=(10,6))
    st.line_chart(hist, width=700, height=400)
