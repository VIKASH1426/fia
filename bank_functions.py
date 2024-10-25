import pickle

# Load the bank accounts from a pickle file
try:
    with open('bank_accounts.pkl', 'rb') as f:
        bank_accounts = pickle.load(f)
except (FileNotFoundError, EOFError):
    bank_accounts = {}


# Save the bank accounts to a pickle file
def save_bank_accounts():
    with open('bank_accounts.pkl', 'wb') as f:
        pickle.dump(bank_accounts, f)


# Bank account management functions
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


# Cash reserve calculation
def calculate_cash_reserve():
    total_bank_balance = sum(bank_accounts.values())
    suggested_reserve = 0.30 * total_bank_balance  # 30% of the total balance
    min_reserve = 10000  # Minimum reserve amount
    return suggested_reserve, min_reserve


# Function to get excess cash (money beyond reserve)
def get_excess_cash():
    total_bank_balance = sum(bank_accounts.values())
    suggested_reserve, min_reserve = calculate_cash_reserve()
    excess_cash = total_bank_balance - max(suggested_reserve, min_reserve)
    return excess_cash if excess_cash > 0 else 0


# Bond Recommendation Function
def recommend_bonds(amount):
    bonds = {
        "U.S. Treasury Bond ETF (TLT)": "Low-risk bond backed by the U.S. government",
        "Corporate Bond ETF (LQD)": "Investment-grade corporate bonds",
        "Municipal Bond ETF (MUB)": "Tax-free municipal bonds"
    }

    st.write(f"You have ${amount:.2f} available for bond investments.")
    st.write("Based on your available funds, here are some bond ETFs to consider:")

    for bond, description in bonds.items():
        st.write(f"{bond}: {description}")



