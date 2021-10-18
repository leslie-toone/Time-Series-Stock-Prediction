####################################################################################
# Use exponential smoothing to see if it can pick up a better trend than moving average
####################################################################################

# originally based on code found at
# https://colab.research.google.com/drive/1tebTeNCNPhcQX9SLEP62lebRWDlMea2d?usp=drive_open
# https://www.youtube.com/watch?v=Rr-ztgKuaSA
# and
# https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775

#before running this code, you must first create the data set using
#C:\Users\lesli\pythonProject2\Stock Prediction Models\Stock Prediction_Data Set Creation.py

#Use the following to explore the data set
#C:\Users\lesli\pythonProject2\Stock Prediction Models\Time Series _Moving Average_Dickey Fuller_PACF and ACF_Seasonal Plots.py

''' simple exponential smoothing (SES) forecast is somewhat superior to the simple
moving average (SMA) forecast because it places relatively more weight on the most recent observation'''
'''
https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/
Time series methods like the Box-Jenkins ARIMA family of methods develop a model where the 
prediction is a weighted linear sum of recent past observations or lags. 
Exponential smoothing forecasting methods are similar in that a prediction is a weighted 
sum of past observations, but the model explicitly uses an exponentially decreasing weight for past observations.

Exponential smoothing uses a similar logic to moving average, but less importance is given 
to observations as we move further from the present.
alpha is a smoothing factor that takes values between 0 and 1. It determines how fast the 
weight decreases for previous observations.
the smaller the smoothing factor, the smoother the time series will be. This makes sense, 
because as the smoothing factor approaches 0, we approach the moving average model.'''


'''Double exponential smoothing is used when there is a trend in the time series. In that case, we use this technique, 
which is simply a recursive use of exponential smoothing twice.
Here, beta is the trend smoothing factor, and it takes values between 0 and 1.'''


from statsmodels.tsa.api import Holt #use this method to avoid FutureWarning when using fit
import warnings
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
sns.set()

# make sure date is the index by specifying index_col=0
stock_hx_data = pd.read_csv(r'C:\Users\lesli\pythonProject2\Stock Prediction Models\data\stock_hx_data.csv',
                            header=[0, 1], index_col=0)

stock_hx_data.sort_values('Date', inplace=True)

print(stock_hx_data.shape, type(stock_hx_data))

# note that weekends are not included in this data set since the market isn't open, and data is already sorted by date

tck = np.unique([second for first, second in stock_hx_data.columns])

# look into https://towardsdatascience.com/time-series-in-python-part-2-dealing-with-seasonal-data-397a65b74051

''' 
In fit1 we again choose not to use the optimizer and provide explicit values for alpha and beta

 https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html'''
def double_exponential_smoothing(ticker, series, alpha, beta):
    fit1=Holt(series, initialization_method='estimated').fit(smoothing_level=alpha,smoothing_trend=beta,optimized=False)
    fcast1=fit1.forecast(1)
    return fit1, fcast1

# Split Data into train, test, and validation datasets
for i in tck:
    test_percent = 0.33
    no_test_obs = int(np.round(test_percent * len(stock_hx_data['Adj Close'])))
    # can slice the data frame on the keys
    # this is helpful
    # https://stackoverflow.com/questions/45128523/pandas-multiindex-how-to-select-second-level-when-using-columns
    data=stock_hx_data.loc[:, (slice(None), i)]
    data=data.droplevel(1,axis=1)
    train = data[:-no_test_obs]
    testing = data[-no_test_obs:]

    # breaking the testing data into validation(helps tune model) and test set (used to calculate final model unbiased
    # accuracy )
    validation_percent = 0.33
    num_validation_obs = int(np.round(validation_percent * len(testing)))
    # Negative indexing starts with the end so you don't need to know the length of list
    # First index is inclusive (before the :) and last (after the :) is not
    # say no_validation_obs=50, this would take the last 50 observations including observation 50
    test = testing[-num_validation_obs:]
    # this would take everything before the last 50 observations excluding observation 50
    validation = testing[:-num_validation_obs]


    def plot_results(ticker, train_data, test_data, predicted_y, method):
        plt.figure(figsize=(10, 6))
        rmse = round(sqrt(mean_squared_error(test_data, predicted_y)),2)
        print(f'{ticker} {method} (RMSE: {rmse})')
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.plot(fit1.fittedvalues, 'yellow', label='Train Data: Dbl Exp Smoothing')
        plt.plot(train_data, 'green', label='Train Data')
        plt.plot(test_data, 'blue', label='Test Data')
        plt.plot(predicted_y, 'orange', label='Predicted')
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title(f'{ticker} {method} (RMSE: {rmse})')
        plt.show()

    history = [x for x in train['Adj Close']]
    predictions = list()
    gain = list()
    signal = list()
    # recall :
    # Alpha determines how fast the weight decreases for previous observations,the smaller the
    # smoothing factor, the smoother the time series will be(as the smoothing factor approaches 0, we approach
    # the moving average model).
    # beta=trend smoothing factor. if beta is big you see more trend component, small beta smooths that out
    for t in range(len(testing['Adj Close'])):
        fit1, yhat=double_exponential_smoothing(i, history, alpha=0.9, beta=0.02)
        obs = testing['Adj Close'][t]
        gain.append(float(yhat-obs))
        history.append(obs)
        predictions.append(float(yhat))

    #Setting with copy warning: we need to transpose testing and use loc
    Results=testing.T
    Results.loc[('gain')] = np.array(gain).T
    Results.loc[('predictions')] = np.array(predictions).T

    # transpose results back so that we can finish analysis
    testing = Results.T
    testing['Actual_Change']=(testing['Adj Close'].shift(-1)-testing['Adj Close'])
    testing['Change']=np.select([testing['Actual_Change'] < 0, testing['Actual_Change'] > 0], [-1, 1],default=0)
    # if we predict it will increase, then we buy, what is our P/L
    #Errors are normally distributed with zero mean and constant variance which is a good sign.
    plot_results(i,train['Adj Close'], testing['Adj Close'], testing['predictions'], 'Double Exp Smoothing')


    # create Action=0=hold, Action=1=Buy, Action=-1 Sale
    testing['action'] = 0
    testing['action'] = np.select([testing['gain'] < 0, testing['gain'] > 0], [-1, 1],default=0)
    #position will tell us if our action changed, if equal to zero then we are the same
    testing['position'] = testing['action'].diff() / 2
    testing['position'][0] = testing['action'][0]

    # visualize trading signals and position
    fig = plt.figure(figsize=(14, 6))
    # "111" means "1x1 grid
    bx = fig.add_subplot(111)

    # plot two different assets
    l1, = bx.plot(testing['Adj Close'], c='#4abdac')
    u1, = bx.plot(testing['Adj Close'][testing['position'] == 1], lw=0, marker='^', markersize=8, c='g', alpha=0.7)
    d1, = bx.plot(testing['Adj Close'][testing['position'] == -1], lw=0, marker='v', markersize=8, c='r', alpha=0.7)
    l2, = bx.plot(testing['predictions'], c='plum')

    bx.set_ylabel(i, )
    bx.yaxis.labelpad = 15

    bx.set_xlabel('Date')
    plt.xticks(rotation=90,fontsize=8)
    bx.xaxis.labelpad = 15

    plt.legend([l1, u1, d1, l2], [f'{i} Current Adj Close', 'Buy', 'Sell',f'{i} Forcasted Adj Close'],
               loc='upper left')
    plt.title(f'{i} Exponential Smoothing Signals and Position')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout();

    # Portfolio Profit and Loss Calculation

    # Start with an initial capital of 1,000 and calculate the maximum number of shares position for each stock
    # using the initial capital.
    # On any given day, total profit and loss from the first stock will be total holding in that
    # stock and cash position for that stock.
    # Based on the position for the stock 1, we calculate their respective daily returns.

    #initial capital to calculate the actual pnl
    initial_capital = 3000

    portfolio = pd.DataFrame()
    portfolio['Adj Close'] = testing['Adj Close']
    portfolio['action'] = testing['action']
    portfolio['position'] = testing['position']
    hold=list()
    cash=list()
    cost=list()
    trx_amt=list()
    capital=initial_capital
    num_shares=0
    stock_value=0

    for t in range(len(portfolio['Adj Close'])):
        if testing['position'][t] ==1:
            num_shares = initial_capital // portfolio['Adj Close'][t]
            cost = testing['position'][t] * portfolio['Adj Close'][t] * num_shares
            stock_value=cost
        elif testing['position'][t] ==-1:
            cost = testing['position'][t] * portfolio['Adj Close'][t] * num_shares
            num_shares=0
            stock_value=0
        elif testing['position'][t] ==0:
            cost = 0
        capital=capital-cost
        hold.append(stock_value)
        cash.append(capital)
        trx_amt.append(cost)
        # Percentage change between the current and a prior element.
    portfolio['Trx Amt']=trx_amt
    portfolio['holdings']=hold
    portfolio['cash']=cash
    portfolio['total asset'] = portfolio['holdings'] + portfolio['cash']
    portfolio['return'] = ((portfolio['total asset']- initial_capital) / initial_capital) * 100

    # calculate CAGR Compounded Annual Growth Rate (CAGR) for the strategy
    final_portfolio = portfolio['total asset'].iloc[-1]
    delta = (pd.to_datetime(portfolio.index[-1]) - pd.to_datetime(portfolio.index[0])).days
    print('Number of days = ', delta)
    YEAR_DAYS = 365
    #CAGR=(Ending Value/Beginning Value)^(1/# yr) -1 where #yr=delta/365 so 1/#yr=365/delta
    CAGR = ((final_portfolio / initial_capital) ** (YEAR_DAYS / delta)) - 1
    print('CAGR  = {:.3f}%'.format(CAGR * 100))

    total_return = ((portfolio['total asset'][-1] - portfolio['total asset'][0]) / portfolio['total asset'][0]) * 100
    print('Total Return={:.2f}%'.format(total_return))

    # plot the asset value change of the portfolio and pnl
    fig = plt.figure(figsize=(14, 6), )
    ax = fig.add_subplot(111)
    l1, = ax.plot(portfolio['total asset'], c='g')
    ax.set_ylabel('Asset Value')
    ax.yaxis.labelpad = 15
    ax.set_xlabel('Date')
    ax.xaxis.labelpad = 15
    plt.xticks(rotation=90, fontsize=8)
    plt.suptitle(f'{i} Portfolio Performance with Profit and Loss')
    plt.title('Number of days = {}, Total Return={:.2f}%, CAGR  = {:.3f}%'.format(delta,total_return,CAGR * 100,))
    plt.legend([l1], ['Total Portfolio Value'], loc='upper left');

    plt.show()
    print(portfolio)

    # Plot confusion matrices
    pd.crosstab(testing['action'],testing['Change'])
    #Subset print statements
    #print(testing[testing['Change']==0])
    #print(testing['Change'][testing['Change']!=0])
    plt.figure(figsize=(15, 5))

    cm = confusion_matrix(testing['action'][testing['Change']!=0],testing['Change'][testing['Change']!=0])
    df_cm = pd.DataFrame(cm, index=['Decrease', 'Increase'], columns=['Decrease', 'Increase'])
    plt.subplot(121)
    plt.title(f"{i} Confusion Matrix\n")
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n {v2}\n {v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, fmt='', cmap="YlGnBu")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")




########################################################

'''Compound annual growth rate(CAGR) is the rate of return that would be required for an investment to grow from its beginning
balance to it sending balance, assuming the profits were reinvested at the end of each year of the investment's life span.

CAGR seems profitable 19% MSFT however:
We have not accounted for costs related to trading.
There is always a limitation of using historical data to forecast the future.'''

'''Residual Check
   Once we have a fitted model to the data, it is necessary to check the residual plots to verify the validity of the
    model fit. A good forecasting method will yield residuals with the following properties:
   1)The residuals are uncorrelated. If there are correlations between residuals, then there is information
   left in the residuals that should be used in computing forecasts.
   2) The residuals have zero mean. If the residuals have a mean other than zero, then the forecasts are biased.'''



#######################################################################################################################
'''To LOAD saved model 
# load model
loaded = ARIMAResults.load('Stock Prediction Models/sarima_model.pkl')

# predict tomorrow
sarima_model_fit = loaded
sarima_output = sarima_model_fit.forecast()
sarima_yhat = sarima_output[0]

print(sarima_yhat)'''

'''OTHER IDEAS: 1) Look at the weekly differences (arima_model.py)
2) normalize data https://www.youtube.com/watch?v=Rr-ztgKuaSA about 30:00min in
3) look into using timeseries split to split the data
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
'''
# PROBLEMS WHEN WE USED MULTI-INDEX
'''note SettingWithCopyWarning should never be ignored. There are two possible ways to access a subset of a
       DataFrame: either one could create a reference to the original data in memory (a view) or copy the subset into a
       new, smaller DataFrame (a copy). A view is a way of looking at a particular portion the original data, whereas a
       copy is a clone of that data to a new location in memory. As our diagram earlier showed, modifying a view will
       modify the original variable but modifying a copy will not.
       See https://www.dataquest.io/blog/settingwithcopywarning/

       we need to transpose testing and use loc, however it struggles updating values for each ticker
       Results=testing.T
       print(Results)
       #this works but not for each ticker, just for the last one
       Results.loc[('sarima_predictions')] =np.array(sarima_predictions).T
       Results.loc[('gain')] =np.array(gain).T
       Results.loc[('signal')] =np.array(signal).T
       #transpose results back so that we can finish analysis
       Results=Results.T
       #transposing got rid of the multiindex so now I don't know how to calculate error, and I couldn't calculate
       #error before transposing since you can't shift when transposed
       Results['error'] = (Results[('Adj Close', i)]-Results['sarima_predictions'].shift(1))'''
# even though I get a warning, I'm pretty sure it's doing what I want it to do and it's complicated using loc as recommended
'''
#######################################################################################################################
# MULTIINDEX STUDY
# indexing and selecting multiindex https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
# note that .shift(1) will now align the predictions with the day they are predicted for
# (prediction on 30th, shows as the predicted value on the 31st) if .shift(-1)
# then predictions made on the 30th show as the predicted on the 29th

# will print the column index of each level
print(list(testing.columns.levels[0]))  # ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
print(list(testing.columns.levels[1]))  # ['AAPL', 'GOOGL', 'MSFT']

print(testing.index.get_level_values(
    0))  # Index(['2021-05-05', '2021-05-06', '2021-05-07',...'2021-08-31'],dtype='object', name='Date')

print(testing['Adj Close'])  # gives Adj Close for each date for each TCK

print(testing['AAPL'])  # -----DOES NOT WORK

print(testing['Adj Close', 'AAPL'])  # ADJ CLOSE just for AAPL
print(testing["Adj Close"]["AAPL"])  # same as above

# keeps all the defined levels of an index, even if they are not actually used. for example this
# gives[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], ['AAPL', 'GOOGL', 'MSFT']]
# This is done to avoid a recomputation of the levels in order to make slicing highly performant.
print(testing[['Adj Close', 'Close']].columns.levels)
print(testing.index.names)  # Date

# if we transpose testing we can do the following, but it doesn't work on testing without transposing
t = testing.T
print(t)
print(t.loc['Adj Close'])
print(t.loc[('Adj Close', 'AAPL'), "2021-05-05"])'''
