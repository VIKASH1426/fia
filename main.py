import streamlit as st
import pickle
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import os
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# ---- Bank Functions ---- #
def load_bank_accounts():
    try:
        if os.path.exists('bank_accounts.pkl'):
            with open('bank_accounts.pkl', 'rb') as f:
                return pickle.load(f)
        return {}
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        st.warning(f"Error loading bank accounts: {e}")
        return {}

bank_accounts = load_bank_accounts()

def save_bank_accounts():
    try:
        with open('bank_accounts.pkl', 'wb') as f:
            pickle.dump(bank_accounts, f)
    except Exception as e:
        st.error(f"Error saving bank accounts: {e}")

def add_bank_account(account_name, balance):
    bank_accounts[account_name] = balance
    save_bank_accounts()
    return f"Added bank account '{account_name}' with balance ${balance:.2f}."

def remove_bank_account(account_name):
    if account_name in bank_accounts:
        del bank_accounts[account_name]
        save_bank_accounts()
        return f"Removed bank account '{account_name}'."
    else:
        return f"Bank account '{account_name}' not found."

def update_bank_account(account_name, balance):
    if account_name in bank_accounts:
        bank_accounts[account_name] = balance
        save_bank_accounts()
        return f"Updated bank account '{account_name}' to balance ${balance:.2f}."
    else:
        return f"Bank account '{account_name}' not found."

def show_bank_accounts():
    if bank_accounts:
        response = "Your bank accounts:\n"
        for account_name, balance in bank_accounts.items():
            response += f"{account_name}: ${balance:.2f}\n"
        return response.strip()
    else:
        return "No bank accounts found."

def total_balance():
    return f"Total balance across all accounts: ${sum(bank_accounts.values()):.2f}"

# ---- Stock Functions ---- #
def load_portfolio():
    try:
        if os.path.exists('portfolio.pkl'):
            with open('portfolio.pkl', 'rb') as f:
                return pickle.load(f)
        return {}
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        st.warning(f"Error loading portfolio: {e}")
        return {}

portfolio = load_portfolio()

def save_portfolio():
    try:
        with open('portfolio.pkl', 'wb') as f:
            pickle.dump(portfolio, f)
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")

def add_stock(ticker, amount):
    ticker = ticker.upper()
    if ticker in portfolio:
        portfolio[ticker] += amount
    else:
        portfolio[ticker] = amount
    save_portfolio()
    return f"Added {amount} shares of {ticker} to your portfolio."

def remove_stock(ticker, amount):
    ticker = ticker.upper()
    if ticker in portfolio:
        if portfolio[ticker] >= amount:
            portfolio[ticker] -= amount
            if portfolio[ticker] == 0:
                del portfolio[ticker]
            save_portfolio()
            return f"Removed {amount} shares of {ticker} from your portfolio."
        else:
            return "You don't have that many shares."
    else:
        return f"No shares of {ticker} found in your portfolio."

def show_portfolio():
    if portfolio:
        response = "Your current portfolio:\n"
        for ticker, shares in portfolio.items():
            response += f"{ticker}: {shares} shares\n"
        return response.strip()
    else:
        return "Your portfolio is empty."

def portfolio_worth():
    total_value = 0
    response = ""
    for ticker, shares in portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            total_value += shares * price
            response += f"{ticker}: {shares} shares at ${price:.2f} each, total ${shares * price:.2f}\n"
        except IndexError:
            response += f"Could not retrieve data for {ticker}. Data might be unavailable.\n"
        except Exception as e:
            response += f"Error retrieving data for {ticker}: {e}\n"
    response += f"Your portfolio is worth: ${total_value:.2f} USD"
    return response

# Stock Prediction Chart Function
@st.cache_data(ttl=3600)
def stock_prediction_chart(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    if not hist.empty:
        last_price = hist['Close'].iloc[-1]
        st.line_chart(hist['Close'])
        st.write(f"The last price of {ticker} is ${last_price:.2f}.")
    else:
        st.write(f"No historical data available for {ticker}.")

# --- Modern Portfolio Theory Optimization Functions --- #
@st.cache_data(ttl=3600)
def get_stock_data(tickers, period='1y'):
    try:
        df = yf.download(tickers, period=period)['Close']
        return df
    except Exception as e:
        st.error(f"Error retrieving stock data: {e}")
        return None

def optimize_portfolio(cov_matrix, mean_returns, risk_free_rate=0.02, max_weight=0.5, min_weight=0.05):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))

    initial_weights = num_assets * [1. / num_assets]

    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Negative because we are minimizing

    result = minimize(negative_sharpe_ratio, initial_weights, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def display_allocation_pie_chart(tickers, weights):
    fig, ax = plt.subplots()
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl", len(tickers)))
    ax.axis('equal')
    st.pyplot(fig)

def display_risk_return_chart(mean_returns, portfolio_return, portfolio_volatility, optimized_weights):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=mean_returns.index, y=mean_returns.values, size=[w * 100 for w in optimized_weights],
                    hue=mean_returns.values, palette="viridis", legend=False, ax=ax)
    for i, ticker in enumerate(mean_returns.index):
        ax.annotate(ticker, (mean_returns.index[i], mean_returns.values[i]))
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Average Return")
    ax.set_title("Stock Returns vs Weights")
    st.pyplot(fig)

# Enhanced MPT-based Portfolio Optimization Function
def recommend_mpt_allocation():
    st.header("Portfolio Optimization Using Modern Portfolio Theory")

    tickers = list(portfolio.keys())
    if not tickers:
        return st.write("Your portfolio is empty. Please add stocks to proceed with optimization.")

    df = get_stock_data(tickers)
    if df is None:
        return st.write("Error fetching stock data. Please try again later.")

    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    portfolio_weights = np.array([portfolio[ticker] for ticker in tickers])
    portfolio_weights /= portfolio_weights.sum()
    current_portfolio_return, current_portfolio_volatility, current_sharpe_ratio = calculate_portfolio_metrics(
        portfolio_weights, mean_returns, cov_matrix)

    st.subheader("Current Portfolio Performance")
    st.write(f"Expected annual return: **{current_portfolio_return * 100:.2f}%**")
    st.write(f"Portfolio risk (volatility): **{current_portfolio_volatility * 100:.2f}%**")
    st.write(f"Sharpe ratio (return-to-risk): **{current_sharpe_ratio:.2f}**")

    risk_tolerance = st.slider("Select your risk tolerance level (higher = more risk)", 0.1, 1.0, 0.5)

    optimized_result = optimize_portfolio(cov_matrix, mean_returns)
    optimized_weights = optimized_result.x
    optimized_return, optimized_volatility, optimized_sharpe_ratio = calculate_portfolio_metrics(
        optimized_weights, mean_returns, cov_matrix)

    st.subheader("Optimized Portfolio Allocation")
    display_allocation_pie_chart(tickers, optimized_weights)

    st.write("### Recommended Allocation:")
    for ticker, weight in zip(tickers, optimized_weights):
        explanation = ""
        if weight > 0.5:
            explanation = f" - **{ticker}** has a higher weight because it's expected to deliver strong returns based on past performance."
        elif weight > 0.25:
            explanation = f" - **{ticker}** is a balanced pick, with moderate returns and risk, making it ideal for diversification."
        else:
            explanation = f" - **{ticker}** has a smaller allocation due to its lower risk-adjusted return, which helps reduce overall volatility."

        st.write(f"{ticker}: **{weight * 100:.2f}%**{explanation}")

    st.subheader("Performance of Optimized Portfolio")
    st.write(f"Expected annual return: **{optimized_return * 100:.2f}%**")
    st.write(f"Portfolio risk (volatility): **{optimized_volatility * 100:.2f}%**")
    st.write(f"Sharpe ratio (return-to-risk): **{optimized_sharpe_ratio:.2f}**")

    display_risk_return_chart(mean_returns, optimized_return, optimized_volatility, optimized_weights)

    st.write("### Key Differences Between Current and Optimized Portfolios:")
    if current_portfolio_return < optimized_return:
        st.write(f"- The optimized portfolio improves expected returns from **{current_portfolio_return * 100:.2f}%** to **{optimized_return * 100:.2f}%**.")
    if current_portfolio_volatility > optimized_volatility:
        st.write(f"- Risk is reduced from **{current_portfolio_volatility * 100:.2f}%** to **{optimized_volatility * 100:.2f}%**.")
    if current_sharpe_ratio < optimized_sharpe_ratio:
        st.write(f"- Sharpe ratio increases from **{current_sharpe_ratio:.2f}** to **{optimized_sharpe_ratio:.2f}**, indicating better risk-adjusted performance.")
# --- Stock Forecast App (New Section) --- #
def stock_forecast_app():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")


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
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

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
# --- Streamlit Application --- #
def main():
    st.title("Investment Portfolio Management")
    st.markdown("<hr style='border:1px solid #EEE'/>", unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ("Bank Accounts", "Stocks", "Portfolio Optimization", "Stock Forecast App"))

    if options == "Bank Accounts":
        st.header("Manage Bank Accounts")
        action = st.selectbox("Action", ("Add Account", "Remove Account", "Update Account", "Show Accounts"))
        if action == "Add Account":
            account_name = st.text_input("Account Name")
            balance = st.number_input("Balance", min_value=0.0)
            if st.button("Add Account"):
                st.write(add_bank_account(account_name, balance))
        elif action == "Remove Account":
            account_name = st.text_input("Account Name")
            if st.button("Remove Account"):
                st.write(remove_bank_account(account_name))
        elif action == "Update Account":
            account_name = st.text_input("Account Name")
            balance = st.number_input("New Balance", min_value=0.0)
            if st.button("Update Account"):
                st.write(update_bank_account(account_name, balance))
        elif action == "Show Accounts":
            st.write(show_bank_accounts())
            st.write(total_balance())

    elif options == "Stocks":
        st.header("Manage Stock Portfolio")
        action = st.selectbox("Action", ("Add Stock", "Remove Stock", "Show Portfolio", "View Stock Prediction"))
        if action == "Add Stock":
            ticker = st.text_input("Stock Ticker")
            amount = st.number_input("Number of Shares", min_value=0)
            if st.button("Add Stock"):
                st.write(add_stock(ticker, amount))
        elif action == "Remove Stock":
            ticker = st.text_input("Stock Ticker")
            amount = st.number_input("Number of Shares", min_value=0)
            if st.button("Remove Stock"):
                st.write(remove_stock(ticker, amount))
        elif action == "Show Portfolio":
            st.write(show_portfolio())
            st.write(portfolio_worth())
        elif action == "View Stock Prediction":
            ticker = st.text_input("Stock Ticker")
            if st.button("View Chart"):
                stock_prediction_chart(ticker)

    elif options == "Portfolio Optimization":
        st.header("Portfolio Optimization Using Modern Portfolio Theory")
        recommend_mpt_allocation()

    elif options == "Stock Forecast App":
        stock_forecast_app()

if __name__ == "__main__":
    main()
