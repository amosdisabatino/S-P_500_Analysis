import bs4 as bs
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader as web
import pickle
import requests
from sklearn.preprocessing import StandardScaler



def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    print("Compiling data...")
    for ticker in tickers:
        if ticker not in ('BRK.B', 'BF.B', 'FANG', 'DISH', 'ODFL', 'ORCL',
                          'OTIS', 'CTXS'):
            df = pd.read_csv('companies/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)

            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(
                ['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        else:
            continue
    main_df.to_csv('sp_500_data.csv')
    print("Data Compiled!")


def load_prices(reload_tickers=False):

    if reload_tickers:
        tickers = load_sp500_tickers()
    else:
        if os.path.exists('sp500tickers.pickle'):
            with open('sp500tickers.pickle', 'rb') as f:
                tickers = pickle.load(f)

    if not os.path.exists('companies'):
        os.makedirs('companies')

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now()

    for ticker in tickers:
        if ticker not in ('BRK.B', 'BF.B', 'FANG', 'DISH', 'ODFL', 'ORCL',
                          'OTIS', 'CTXS'):
            if not os.path.exists('companies/{}.csv'.format(ticker)):
                print("{} is loading...".format(ticker))
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('companies/{}.csv'.format(ticker))
            else:
                print("{} already downloaded!".format(ticker))
        else:
            continue


def load_sp500_tickers():

    link = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'

    response = requests.get(link)

    soup = bs.BeautifulSoup(response.text, 'lxml')

    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text[:-1]
        tickers.append(ticker)

    with open("sp500tickers.pickle", 'wb') as f:
        pickle.dump(tickers, f)

    return tickers


# load_prices()
# compile_data()

sp500 = pd.read_csv('sp_500_data.csv')

big3_df = pd.DataFrame(sp500[['MSFT', 'AMZN', 'AAPL']])

scaler = StandardScaler()

standard_sp500 = scaler.fit_transform(sp500.loc[:, sp500.columns != 'Date'])
label = list(sp500.loc[:, sp500.columns != 'Date'].keys())
standard_sp500_df = pd.DataFrame(standard_sp500, columns=label)

correlation_normal = sp500.loc[:, sp500.columns != 'Date'].corr()
big3_correlation = big3_df.corr()
std_correlation =\
    standard_sp500_df.corr()
print("Standard Data Correlation: ", std_correlation)
print("Normal Data Correlation: ", correlation_normal)
print("BIG3 Correlation: ", big3_correlation)

plt.matshow(std_correlation)
plt.title("STD Correlation")
plt.show()
plt.matshow(correlation_normal)
plt.title("Normal Correlation")
plt.show()

# Normal Data

y1 = sp500['MSFT']
y2 = sp500['AMZN']
y3 = sp500['AAPL']

# STD Data

y1_std = standard_sp500_df['MSFT']
y2_std2 = standard_sp500_df['AMZN']
y3_std3 = standard_sp500_df['AAPL']

# NOTE: "axis 0" represents rows and "axis 1" represents columns

sp500.set_index('Date', inplace=True)
sp500['mean'] = sp500.mean(axis=1)
mean = sp500['mean']

standard_sp500_df['Date'] = pd.to_datetime(sp500.index)
standard_sp500_df.set_index('Date', inplace=True)
standard_sp500_df['sp500_mean'] = standard_sp500_df.mean(axis=1)
standard_mean = standard_sp500_df['sp500_mean']

# NORMAL DATA PLOT

plt.grid()
plt.plot(standard_sp500_df.index, y1, label="Microsoft")
plt.plot(standard_sp500_df.index, y2, label="Amazon")
plt.plot(standard_sp500_df.index, y3, label="Apple")
plt.plot(standard_sp500_df.index, mean, label="Mean")
plt.legend(loc='upper left')
plt.title("BIG 3 & 'S&P500' Normal Mean")
plt.show()

# STANDARD DATA PLOT

plt.grid()
plt.plot(standard_sp500_df.index, y1_std, label="Microsoft STD")
plt.plot(standard_sp500_df.index, y2_std2, label="Amazon STD")
plt.plot(standard_sp500_df.index, y3_std3, label="Apple STD")
plt.plot(standard_sp500_df.index, standard_mean, label=" STD Mean")
plt.legend(loc='upper left')
plt.title("BIG 3 & 'S&P500' STD Mean")
plt.show()

# STANDARD DATA PLOT & REGRESSION LINE


x_std_reg = standard_mean.index.map(mdates.date2num)
fit = np.polyfit(x_std_reg, standard_mean.values, 1)
fit1d = np.poly1d(fit)

plt.grid()
plt.plot(standard_mean.index, standard_mean.values, 'b')
plt.plot(standard_mean.index, fit1d(x_std_reg), 'r')
plt.title("'S&P500' STD Mean & Regression Line")
plt.show()

# SPECIFY 'TIME FRAME' TO REGRESSION LINE

# Ukraine War

r_start = dt.datetime(2022, 2, 1)
r_end = dt.datetime(2022, 8, 1)

war_data = standard_sp500_df.reset_index()

pos1 = war_data[war_data.Date >= r_start].index[0]  # First Index
pos2 = war_data[war_data.Date <= r_end].index[-1]  # Last Index

war_data = war_data.iloc[pos1:pos2]

dates = war_data.Date.map(mdates.date2num)

fit = np.polyfit(dates, war_data['sp500_mean'], 1)
fit1d = np.poly1d(fit)

# Covid-19

r_start_2 = dt.datetime(2020, 1, 1)
r_end_2 = dt.datetime(2020, 5, 1)

covid_data = standard_sp500_df.reset_index()

pos1_2 = covid_data[covid_data.Date >= r_start_2].index[0]
pos2_2 = covid_data[covid_data.Date <= r_end_2].index[-1]

covid_data = covid_data.iloc[pos1_2:pos2_2]

covid_dates = covid_data.Date.map(mdates.date2num)

fit_2 = np.polyfit(covid_dates, covid_data['sp500_mean'], 1)
fit1d_2 = np.poly1d(fit_2)

plt.grid()
plt.plot(standard_mean.index, standard_mean.values, 'b')
plt.plot(war_data.Date, fit1d(dates), 'r')
plt.plot(covid_data.Date, fit1d_2(covid_dates), 'r')
plt.title("'S&P500' STD AVG, Covid-19 & Ukraine War Regression Line")
plt.show()
