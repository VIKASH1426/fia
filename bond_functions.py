import requests
import pickle

ALPHA_VANTAGE_API_KEY = 'RHDSZ9ZA43C1GXE4'
BOND_API_URL = "https://www.alphavantage.co/query"


def fetch_bond_yield():
    params = {
        "function": "BOND_YIELD",
        "symbol": "IN10Y",  # Indian 10-Year bond
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(BOND_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            bond_yield = float(data['Bond Yield'])
            return bond_yield
        except KeyError:
            return None
    return None


def recommend_bonds(amount):
    bonds = {
        "U.S. Treasury Bond ETF (TLT)": "Low-risk bond backed by the U.S. government",
        "Corporate Bond ETF (LQD)": "Investment-grade corporate bonds",
        "Municipal Bond ETF (MUB)": "Tax-free municipal bonds"
    }

    indian_bond_yield = fetch_bond_yield()

    print(f"You have ${amount:.2f} for bond investments.")
    print("Recommended bonds based on yields:")

    if indian_bond_yield:
        print(f"Indian 10-Year Government Bond Yield: {indian_bond_yield:.2f}%")
        if indian_bond_yield > 6:
            print(f"Consider more Indian bonds since yield is {indian_bond_yield:.2f}%.")
        else:
            print(f"Lower yields at {indian_bond_yield:.2f}%, consider alternatives.")

    for bond, description in bonds.items():
        print(f"{bond}: {description}")

