
import pandas as pd
import yfinance as yf
import datetime

#tips on importing data from yfinance https://algotrading101.com/learn/yfinance-guide/
end = datetime.date(2021, 8, 30)
days = datetime.timedelta(365)
start = end - days
#print(end, start)

cachedFolderName = 'yahooData/'
dataSetId = 'Stock_Prediction'
tickers_list = ['AAPL', 'MSFT', 'GOOGL']
tickers_data={}

#look at what info is available for each ticker
aapl = yf.Ticker("aapl")
#print(aapl.info.keys())
'''
PROFIT MARGIN:
Both gross profit margin and net profit margin are used to determine how efficient a company's management is in earning profits.
The gross profit margin is calculated by deducting from the revenue the costs associated with the production, such as parts and packaging.
The net profit margin is the bottom line of a company in percentage terms and is the ultimate measure of profitability for a company
(deducts operating expenses and any other expenses, such as debt.)

Return on assets (ROA) is an indicator of how profitable a company is relative to its total assets.

Beta is a measure of the volatility—or systematic risk—of a security or portfolio compared to the market as a whole. 

'52WeekChange' is the data point includes the lowest and highest price at which a stock has traded during the previous 52 weeks.
Investors use this information as a proxy for how much fluctuation and risk they may have to endure over the course of 
a year should they choose to invest in a given stock. 

Trailing P/E is calculated by dividing the current market value, or share price, by the earnings per share over the 
previous 12 months. The forward P/E ratio estimates a company's likely earnings per share for the next 12 months.
'''

keys_to_extract = ['symbol','shortName','country','sector','industry','market','marketCap', 'exchange','currentPrice',
                   'previousClose',  'volume','volume24Hr','averageDailyVolume10Day','fiftyDayAverage',
                   'trailingPE','forwardPE', 'profitMargins', 'grossMargins','freeCashflow','revenueGrowth',
                   'earningsGrowth','returnOnAssets','debtToEquity','totalRevenue','beta3Year','52WeekChange',
                   'revenueQuarterlyGrowth', 'heldPercentInstitutions',
                   'threeYearAverageReturn','fiveYearAverageReturn','dividendRate','dividendYield','lastDividendValue',
                   'trailingAnnualDividendYield',
                   'payoutRatio']


'''We then loop through the list of the tickers, in each case adding to our dictionary a key, 
value pair where the key is the ticker and the value the dataframe returned by the 
info() method for that ticker:'''
for ticker in tickers_list:
    ticker_object = yf.Ticker(ticker)
    ticker_subset={key: ticker_object.info[key] for key in keys_to_extract}
    #convert info() output from dictionary to dataframe
    temp = pd.DataFrame.from_dict(ticker_subset, orient="index")
    temp.reset_index(inplace=True)
    temp.columns = ["Attribute", "Recent"]

    # add (ticker, dataframe) to main dictionary
    tickers_data[ticker] = temp

tickers_data

'''We then combine this dictionary of dataframes into a single dataframe:'''
stock_info = pd.concat(tickers_data)
stock_info = stock_info.reset_index()
stock_info

#Comparing by a particular attribute

previousClose = stock_info[stock_info['Attribute']=='previousClose'].reset_index()
del previousClose["index"] # clean up unnecessary column

#Note: Level_0=ticker, Level_1 variable index, Attribute=level_1 name, Recent: the value of of the attribute
previousClose

#save data to a CSV
stock_info.to_csv(r"C:\Users\lesli\pythonProject2\Stock Prediction Models\data\stock_info.csv")

#bring in historical data
stock_hx_data = yf.download(cachedFolderName=cachedFolderName,
                   dataSetId=dataSetId,
                   tickers=tickers_list,start=start,
                   endD=end,
                   event='history')


print(stock_hx_data.head())
#see columns in the dataframe
print(list(stock_hx_data))

stock_hx_data.to_csv(r"C:\Users\lesli\pythonProject2\Stock Prediction Models\data\stock_hx_data.csv")
