import streamlit as st
from stock_functions import recommend_stocks
from bank_functions import recommend_bonds, calculate_cash_reserve, get_excess_cash


# Function to recommend an investment strategy based on user preferences
def recommend_investment_strategy(invest_percent, risk_tolerance):
    st.subheader("Investment Strategy Recommendations")

    excess_cash = get_excess_cash()
    amount_to_invest = invest_percent * excess_cash

    st.write(
        f"You have chosen to invest {invest_percent * 100:.1f}% of your excess cash reserve, totaling ${amount_to_invest:.2f}.")

    # Risk Tolerance Logic
    if risk_tolerance == "Low":
        st.write(
            "You've selected a **low-risk** strategy. A larger portion of your funds will be directed towards bonds and secure assets.")
        recommend_bonds(amount_to_invest)
    elif risk_tolerance == "Medium":
        st.write("You've selected a **medium-risk** strategy. Your investment will be split between stocks and bonds.")

        # Recommend 50% in bonds, 50% in stocks
        amount_bonds = amount_to_invest * 0.5
        amount_stocks = amount_to_invest * 0.5
        st.write(f"${amount_bonds:.2f} will be invested in bonds:")
        recommend_bonds(amount_bonds)

        st.write(f"${amount_stocks:.2f} will be invested in stocks:")
        stock_recommendations = recommend_stocks()
        st.write("\n".join(stock_recommendations))
    elif risk_tolerance == "High":
        st.write(
            "You've selected a **high-risk** strategy. Most of your funds will be directed towards stocks with higher growth potential.")

        # Recommend 70% in stocks, 30% in bonds
        amount_stocks = amount_to_invest * 0.7
        amount_bonds = amount_to_invest * 0.3
        st.write(f"${amount_stocks:.2f} will be invested in stocks:")
        stock_recommendations = recommend_stocks()
        st.write("\n".join(stock_recommendations))

        st.write(f"${amount_bonds:.2f} will be invested in bonds:")
        recommend_bonds(amount_bonds)
    else:
        st.write("Please select a valid risk tolerance option.")


# Function to get user input and provide the final investment recommendation
def investment_recommendation_section():
    st.subheader("Customized Investment Recommendation")

    # Get user input for percentage of cash reserve to invest
    invest_percent = st.number_input("What percentage of your excess cash reserve would you like to invest?",
                                     min_value=0.0, max_value=100.0,
                                     value=50.0) / 100.0  # Convert percentage to decimal

    # Get user risk tolerance input
    risk_tolerance = st.radio("Select your risk tolerance level:", ["Low", "Medium", "High"])

    # Provide investment recommendation
    if st.button("Get Investment Recommendation"):
        recommend_investment_strategy(invest_percent, risk_tolerance)


# To run this section in the main app
if __name__ == "__main__":
    investment_recommendation_section()
