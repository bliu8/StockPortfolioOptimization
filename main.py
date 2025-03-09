import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import datetime
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value
import pandas_datareader.data as web

start = datetime.datetime(2024, 1, 1)
end = datetime.datetime(2024, 12, 3)

ff_data = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)
factors = ff_data[0][['Mkt-RF', 'SMB', 'HML']] / 100

tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ASML", "TSM", "AMZN", "ORCL", "ADBE", "CRM", "CSCO", "INTC", "AMD",
    "LLY", "JNJ", "PFE", "MRK", "ABBV", "ABT", "TMO", "AMGN", "GILD", "BMY", "CVS", "UNH", "ANTM", "HUM", "CI",
    "BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "SPGI", "MMC", "MSCI", "ICE", "CME",
    "XOM", "CVX", "COP", "SLB", "MPC", "PSX", "BKR", "BP", "SHEL", "TOT", "ENB", "EOG", "PXD", "KMI", "WMB",
    "BA", "CAT", "GE", "MMM", "HON", "LMT", "UNP", "UPS", "FDX", "DE", "RTX", "NOC", "GD", "CSX", "NSC",
    "TSLA", "TM", "GM", "F", "HMC", "VWAGY", "HD", "LOW", "NKE", "SBUX", "MCD", "TGT", "DG", "ROST", "TJX",
    "KO", "PEP", "PG", "WMT", "COST", "MDLZ", "PM", "MO", "CL", "KMB", "GIS", "K", "SYY", "KR", "WBA",
    "VZ", "T", "TMUS", "DIS", "NFLX", "CHTR", "CMCSA", "SPOT", "LYV", "TTWO", "EA", "ATVI", "VIAC", "DISCA", "FOX",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ES", "PEG", "ED", "EIX", "PPL", "WEC", "AWK",
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "DLR", "O", "WELL", "VTR", "AVB", "EQR", "ESS", "MAA", "UDR",
    "LIN", "APD", "SHW", "ECL", "PPG", "NUE", "DOW", "FCX", "MLM", "VMC", "IFF", "ALB", "LYB", "DD", "EMN",
    "V", "MA", "PYPL", "SQ", "FIS", "FISV", "ADP", "INTU", "IBM", "TXN", "QCOM", "MU", "ADI", "LRCX", "AMAT"
]

data = yf.download(tickers, start=start, end=end)['Adj Close']
data.dropna(axis=1, how='all', inplace=True)
returns = data.pct_change().dropna()

def calculate_ff_sensitivities(stock_ticker):
    stock_data = yf.download(stock_ticker, start=start, end=end)['Adj Close'].pct_change().dropna()
    aligned_data = pd.concat([stock_data, factors], axis=1).dropna()
    X = sm.add_constant(aligned_data[['Mkt-RF', 'SMB', 'HML']])
    y = aligned_data[stock_ticker]
    model = sm.OLS(y, X).fit()
    return model.params

expected_returns_ff = {}
risk_free_rate = 0.015 / 252

for stock in returns.columns:
    try:
        ff_coefficients = calculate_ff_sensitivities(stock)
        beta = ff_coefficients['Mkt-RF']
        sensitivity_smb = ff_coefficients['SMB']
        sensitivity_hml = ff_coefficients['HML']
        expected_return = (
            risk_free_rate +
            beta * factors['Mkt-RF'].mean() +
            sensitivity_smb * factors['SMB'].mean() +
            sensitivity_hml * factors['HML'].mean()
        )
        expected_returns_ff[stock] = ((1 + expected_return) ** 252) - 1
    except Exception as e:
        print(f"Error calculating expected return for {stock}: {e}")

risk = returns.std() * np.sqrt(252)
daily_absolute_changes = data.diff().abs().dropna()
sum_absolute_changes = daily_absolute_changes.sum()
average_price = data.mean()
normalized_abs_change = sum_absolute_changes / average_price

results = pd.DataFrame({
    'Expected Return (FF)': pd.Series(expected_returns_ff),
    'Risk (Standard Deviation)': risk,
    'Normalized Absolute Change': normalized_abs_change
}).dropna()

total_capital = 1000
max_allocation = 0.30 * total_capital
min_allocation = 0.005 * total_capital

allocations = {
    stock: LpVariable(f"alloc_{stock}", 0, max_allocation)
    for stock in results.index
}

max_risk = 0.70
min_stocks_to_invest = 10
risk_preference = 0.7

problem = LpProblem("Portfolio_Optimization", LpMaximize)

selected = {
    stock: LpVariable(f"select_{stock}", 0, 1, cat='Binary')
    for stock in results.index
}

expected_return_component = lpSum(
    allocations[stock] * results.loc[stock, 'Expected Return (FF)'] for stock in results.index
)

risk_component = lpSum(
    allocations[stock] * results.loc[stock, 'Risk (Standard Deviation)'] for stock in results.index
)

volatility_component = lpSum(
    allocations[stock] * results.loc[stock, 'Normalized Absolute Change'] for stock in results.index
)

problem += expected_return_component - (risk_preference * risk_component + (1 - risk_preference) * volatility_component)

problem += lpSum(allocations[stock] for stock in results.index) == total_capital

weighted_risk = lpSum(
    allocations[stock] * results.loc[stock, 'Risk (Standard Deviation)'] for stock in results.index
) / total_capital
problem += weighted_risk <= max_risk

for stock in results.index:
    problem += allocations[stock] >= min_allocation * selected[stock]
    problem += allocations[stock] <= max_allocation * selected[stock]

problem += lpSum(selected[stock] for stock in results.index) >= min_stocks_to_invest

status = problem.solve()

if LpStatus[status] != 'Optimal':
    print("Optimization did not find an optimal solution.")
else:
    print("Optimal Allocations:")
    total_invested = 0
    for stock in results.index:
        alloc = value(allocations[stock])
        total_invested += alloc
        if alloc > 0:
            print(f"{stock}: ${alloc:,.2f}")

    expected_return_value = sum(
        value(allocations[stock]) * results.loc[stock, 'Expected Return (FF)']
        for stock in results.index
    )
    risk_value = sum(
        value(allocations[stock]) * results.loc[stock, 'Risk (Standard Deviation)']
        for stock in results.index
    ) / total_capital

    print(f"\nTotal Invested Capital: ${total_invested:,.2f}")
    print(f"Expected Portfolio Return: ${expected_return_value:.2f}")
    print(f"Portfolio Risk (Approximate Weighted Average): {risk_value:.4f}")
