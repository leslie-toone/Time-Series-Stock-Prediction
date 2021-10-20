# originally based on code found at
# https://colab.research.google.com/drive/1tebTeNCNPhcQX9SLEP62lebRWDlMea2d?usp=drive_open
# https://www.youtube.com/watch?v=Rr-ztgKuaSA
# and
# https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775

# before running this code, you must first create the data set using
# C:\Users\lesli\pythonProject2\Stock Prediction Models\Stock Prediction_Data Set Creation.py
"""MORE ON MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)
https://stats.stackexchange.com/questions/299712/what-are-the-shortcomings-of-the-mean-absolute-percentage-error-mape
MAPE DOESN'T MAKE SENSE IF ONE OBSERVED VALUE IS 0, OR CLOSE TO 0 SINCE IT WILL=INFINITY (FOR STATIONARY DIFFERENCES
IT'S OFF), BUT IT MAKES SENSE IF WE DON'T LOOK AT THE DIFFERENCED DATA
RMSE GIVES HIGH WEIGHT TO LARGE ERRORS SINCE SQUARED BEFORE AVERAGING THIS IS OK IF BEING OFF BY 10 FOR EXAMPLE IS TWICE
AS BAD AS BEING OFF BY 5,
MAE (MEAN ABSOLUTE ERROR) IS MORE INTERPRETABLE - JUST THE AVERABE ABSOLUTE ERROR"""

import warnings
from math import sqrt

import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
sns.set()

# make sure date is the index by specifying index_col=0
stock_hx_data = pd.read_csv(r'C:\Users\lesli\pythonProject2\Stock Prediction Models\data\stock_hx_data.csv',
                            header=[0, 1], index_col=0)

stock_hx_data.sort_values('Date', inplace=True)

print(stock_hx_data.shape, type(stock_hx_data))

# note that weekends are not included in this data set since the market isn't open, and data is already sorted by date

# Inspect the data
print(list(stock_hx_data))

# can slice the data frame on the keys
# this is helpful
# https://stackoverflow.com/questions/45128523/pandas-multiindex-how-to-select-second-level-when-using-columns
stock_hx_data.keys()
# stock_hx_data['Adj Close','AAPL']
# stock_hx_data.loc[:, (slice(None), "AAPL")]

tck = np.unique([second for first, second in stock_hx_data.columns])
print(tck)


# tck = ['AAPL']

def mean_absolute_percentage_error(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


########################################################################################################
'''Time Series Complete Guide: 
https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775'''

'''
Exploratory Data Analysis:
Let’s use the moving average model to smooth our time series. 
https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/
Moving average smoothing is a naive and effective technique in time series forecasting.
It can be used for data preparation, feature engineering, and even directly for making predictions.
- The moving average can be used as a source of new information when modeling a time series forecast as a supervised 
learning problem.
In this case, the moving average is calculated and added as a new input feature used to predict the next time step.
- The moving average value can also be used directly to make predictions.
It is a naive model and assumes that the trend and seasonality components of the time series have already been removed 
or adjusted for.'''

# For Confidence intervals, recall that the value of 1.96 is based on the fact that 95% of the area of a normal
# distribution is within 1.96 standard deviations of the mean
#######NON-STATIONARY#################################################################################
'''def plot_moving_average(axes, series, window, ticker, plot_intervals=False, scale=1.96):
    rolling_mean = series.rolling(window=window).mean()
    # subtract 1 since rolling counts # values, but series # starts at 0
    series = series[window - 1:]
    # Drop the missing values from the rolling mean df as well as the series that corresponded to the missing #
    # rolling# mean indexes
    date_list = pd.to_datetime(rolling_mean[rolling_mean.isnull()].index.tolist())
    roll_mean_drop_nan = rolling_mean.dropna()
    series_drop_nan = series[~series.index.isin(date_list)]
    plt.suptitle('%s Moving Average' % ticker)
    axes.set_title('Window: %s' % window)
    # axes.set_title('window size = {}'.format(window))
    axes.plot(roll_mean_drop_nan, 'g', label='Rolling Mean Trend')
    # Plot confidence intervals for smoothed values mean(x) ± z* σ / (√n)
    if plot_intervals:
        # mean absolute error (MAE) is a measure of errors between paired obs, such as predicted versus observed
        mae = round(mean_absolute_error(series_drop_nan, roll_mean_drop_nan),2)
        mape = round(mean_absolute_percentage_error(series_drop_nan, roll_mean_drop_nan),2)
        rmse = round(sqrt(mean_squared_error(series_drop_nan, roll_mean_drop_nan)),2)
        deviation = np.std(series_drop_nan - roll_mean_drop_nan)
        lower_bound = roll_mean_drop_nan - (mae + scale * deviation)
        upper_bound = roll_mean_drop_nan + (mae + scale * deviation)
        axes.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        axes.plot(lower_bound, 'r--')
        print(ticker, window,'MAE: ',mae, 'MAPE: ', mape, 'RMSE: ', rmse)
    axes.plot(series_drop_nan, label='Actual values')
    plt.legend(loc='best')
    axes.grid(True)
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex=False, sharey=True)
for i in tck:
    # for 5 days, you can hardly see a trend, too close to the actual curve
    plot_moving_average(axs[0], stock_hx_data['Adj Close', i], 5, i, plot_intervals=True)
    # Trends over 30 or 90 days are easier to spot
    # Smooth by the previous month (30 days)
    plot_moving_average(axs[1], stock_hx_data['Adj Close', i], 30, i, plot_intervals=True)
    # Smooth by previous quarter (90 days)
    plot_moving_average(axs[2], stock_hx_data['Adj Close', i], 90, i, plot_intervals=True)
    plt.show()
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex=False, sharey=True)
#Notice that for GOOGL the rolling 90 day mean is way below the observed value, this is because google increases so 
#much in 90 days that the average from day 1 to day 90 is way below the observed value at 90 
'''
###################################################################################
# Since the data is not stationary, instead of looking at the moving average on the actual closing prices,
# lets look at the moving averages based on the differences in closing price
#########################################################################################
'''here we are using "diff", but if you wanted to specify the number of lags, you could use
diff = (series - series.shift(numLags))'''


def plot_moving_average_difference(axes, series, window, ticker, plot_intervals=False, scale=1.96):
    # subtract 1 since rolling counts # values, but series # starts at 0
    series = series[window - 1:].to_frame()
    diff = series.diff().squeeze()

    diff.dropna(inplace=True)
    # set index to a date format
    diff.index = pd.to_datetime(diff.index)
    rolling_mean_diff = diff.rolling(window=window).mean()

    # convert dataframe to series so we can subtract (IMPORTANT TO SET ROLLING_MEAN_DIFF= ...)
    rolling_mean_diff = rolling_mean_diff.squeeze()
    date_list = pd.to_datetime(rolling_mean_diff.shift(1)[rolling_mean_diff.shift(1).isna()].index.tolist())
    # print('Dates where Rolling Mean is NA', date_list)
    roll_mean_diff_drop_nan = rolling_mean_diff.dropna()

    # notice that the mean for the observed difference, includes the observed difference,
    # since we want to predict the future, we'll need to shift the predictions and drop those that are missing
    y_hat = roll_mean_diff_drop_nan.shift(1).dropna()

    # print(diff[-20:], rolling_mean_diff[-20:], y_hat)
    diff_drop_nan = diff[~diff.index.isin(date_list)]
    # print('Length of y_hat', len(y_hat), 'Length of diff_drop_nan',len(diff_drop_nan))
    # print(type(diff_drop_nan))
    # convert dataframe to series
    diff_drop_nan = diff_drop_nan.squeeze()
    plt.suptitle('%s Difference Moving Average' % ticker)
    axes.set_title('Window: %s' % window)
    axes.plot(diff_drop_nan, label='Actual Difference')
    axes.plot(y_hat, 'g', label='Predicted Difference')
    # Plot confidence intervals for smoothed values mean(x) ± z* σ / (√n)
    if plot_intervals:
        # mean absolute error (MAE) is a measure of errors between paired obs, such as predicted versus observed
        mae = round(mean_absolute_error(diff_drop_nan, y_hat), 2)
        mape = round(mean_absolute_percentage_error(diff_drop_nan, y_hat), 2)
        rmse = round(sqrt(mean_squared_error(diff_drop_nan, y_hat)), 2)
        deviation = np.std(diff_drop_nan - y_hat)
        lower_bound = y_hat - (mae + scale * deviation)
        upper_bound = y_hat + (mae + scale * deviation)
        axes.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        axes.plot(lower_bound, 'r--')
        print(ticker, window, 'MAE: ', mae, 'MAPE: ', mape, 'RMSE: ', rmse)
    # Use the pyplot interface to change just one subplot (so that we rotate the dates on xaxis)
    plt.sca(axes)
    plt.xticks(rotation=90, fontsize=8)
    axes.grid(True)
    return diff_drop_nan, y_hat


def create_confusion_matrix(i, window, diff_drop_nan, y_hat, axes,fig):
    # Plot confusion matrices
    Observed_Diff = np.select([diff_drop_nan < 0, diff_drop_nan > 0], [-1, 1], default=0)
    Predicted_Diff = np.select([y_hat < 0, y_hat > 0], [-1, 1], default=0)
    print(pd.crosstab(Observed_Diff[Observed_Diff != 0], Predicted_Diff[Observed_Diff != 0]))
    cm = confusion_matrix(Observed_Diff[Observed_Diff != 0], Predicted_Diff[Observed_Diff != 0])
    df_cm = pd.DataFrame(cm, index=['Decrease(-)', 'Increase(+)'], columns=['Decrease(-)', 'Increase(+)'])
    fig.suptitle('Moving Average Confusion Matrix')
    axes.set_title(f"{i} {window} \n")
    group_names = ["True -", "False +", "False -", "True +"]
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n {v2}\n {v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(df_cm, annot=labels, fmt='', cmap="YlGnBu", ax=axes)
    fig.text(0.5, 0.04, "Observed", ha='center')
    fig.text(0.04, 0.5, "Predicted", va='center', rotation='vertical')
    # plt.ylabel()
    # plt.xlabel()
    plt.show()


fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex=False, sharey=True)
fig2, axs2 = plt.subplots(1, 3, figsize=(12, 6), sharex=False, sharey=True)
a=0
for i in tck:
    # for 5 days, you can hardly see a trend, too close to the actual curve
    diff_drop_nan_5, y_hat_5 = plot_moving_average_difference(axs[0], stock_hx_data['Adj Close', i], 5, i,
                                                              plot_intervals=True)
    # Trends over 30 or 90 days are easier to spot
    # Smooth by the previous month (30 days)
    diff_drop_nan_30, y_hat_30 = plot_moving_average_difference(axs[1], stock_hx_data['Adj Close', i], 30, i,
                                                                plot_intervals=True)
    # Smooth by previous quarter (90 days)
    diff_drop_nan_90, y_hat_90 = plot_moving_average_difference(axs[2], stock_hx_data['Adj Close', i], 90, i,
                                                                plot_intervals=True)
    plt.legend(loc="best")
    plt.show()
    ''''create_confusion_matrix(i, 5, diff_drop_nan_5, y_hat_5, axs[0])
    create_confusion_matrix(i, 30, diff_drop_nan_30, y_hat_30, axs[1])
    create_confusion_matrix(i, 90, diff_drop_nan_90, y_hat_90, axs[2])'''
    create_confusion_matrix(i, 90, diff_drop_nan_90, y_hat_90, axs2[a],fig2)
    a = a + 1
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex=False, sharey=True)


#############################################################
# we must turn our series into a stationary process in order to model it.
# Let’s apply the Dickey-Fuller test to see if it is a stationary process:

def tsplot(tck, y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        plt.figure()
        # fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('%s Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value) % tck)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# note that taking the first difference does not necessarily make the data stationary.
stock_diff = stock_hx_data['Adj Close'] - stock_hx_data['Adj Close'].shift(1)
stock_diff = stock_diff[1:]  # drops NAN row so we don't get errors

for i in tck:
    # By the Dickey-Fuller test, the time series is unsurprisingly non-stationary.
    # Also, looking at the autocorrelation plot, we see that it is very high, and
    # it seems that there is no clear seasonality (When data are seasonal, the autocorrelations
    # will be larger for the seasonal lags (at multiples of the seasonal frequency) than for other lags.).
    tsplot(i, stock_hx_data['Adj Close', i], lags=30)
    # Take the first difference to remove the high autocorrelation and to make the process stationary
    tsplot(i, stock_diff[i], lags=30)

# Perform Dickey-Fuller test
# The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary
# (has some time-dependent structure).
# The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
# Awesome! Our series is now stationary (Dickey-Fuller p=0.0000) and we can start modelling!


# Making autocorrelation and partial autocorrelation charts help us choose hyperparameters for the ARIMA(p,d,q) model.

# ACF plot: gives us a measure of how much each "y" value is correlated to the previous n "y" values prior.
# If the ACF value (ARIMA parameter q) is outside of the blue zone, it is a significant correlation, in our case 1

# PACF plot: is the partial correlation function gives us (a sample of) the amount of correlation between two "y"
# values separated by n lags excluding the impact of all the "y" values in between them. In other words,
# it finds correlation of the residuals (which remains after removing the effects which are already explained by the
# earlier lag(s)) with the next lag value hence ‘partial’ and not ‘complete’ as we remove already found variations
# before we find the next correlation. So if there is any hidden information in the residual which can be modeled by
# the next lag, we might get a good correlation and we will keep that next lag as a feature while modeling. Remember
# while modeling we don’t want to keep too many features which are correlated as that can create multicollinearity
# issues. Hence we need to retain only the relevant features. To pick parameter p in the ARIMA, use PACF and choose p
# outside of shaded area, in our case 1

# More about ACF and PACF
# https://towardsdatascience.com/significance-of-acf-and-pacf-plots-in-time-series-analysis-2fa11a5d10a8

# Seasonal Plots for day of the week
'''https://towardsdatascience.com/time-series-analysis-with-theory-plots-and-code-part-1-dd3ea417d8c4
In the seasonal plot we can instantly see:
-More clearly the seasonal pattern if it exists.
-Identify the years in which the pattern changes.
-Identify large jumps or drops.

In the trend and seasonality plots we can see:
-More clearly the trend and the seasonality. 
-outliers.
-Compare years or months easier.'''

stock_diff.index = pd.to_datetime(stock_diff.index)
stock_diff['month'] = stock_diff.index.month.astype("category")
stock_diff['dayofweek'] = stock_diff.index.dayofweek.astype("category")
dayOfWeek = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
stock_diff['weekday'] = stock_diff['dayofweek'].map(dayOfWeek)
# print(stock_diff['dayofweek'], stock_diff['weekday'])
# print(stock_diff.keys())

for i in tck:
    fig, ax = plt.subplots(figsize=(15, 6))
    # palette = sns.color_palette("ch:2.5,-.2,dark=.3",stock_diff['dayofweek'].nunique())
    sns.lineplot(x=stock_diff['month'], y=stock_diff[i], hue=stock_diff['weekday'],
                 ci=None)  # , palette=palette leave this off
    ax.set_title('%s Seasonal Plot\n Difference in Closing Price by Day of the Week' % i, fontsize=20, loc='center',
                 fontdict=dict(weight='bold'))
    ax.set_xlabel('Month', fontsize=16, fontdict=dict(weight='bold'))
    ax.set_ylabel('Adj Close', fontsize=16, fontdict=dict(weight='bold'))
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(x=stock_diff['weekday'], y=stock_diff[i])
    ax.set_title('%s Day-wise Box Plot\n(The Trend)' % i, fontsize=20, loc='center', fontdict=dict(weight='bold'))
    ax.set_xlabel('Day', fontsize=16, fontdict=dict(weight='bold'))
    ax.set_ylabel('Difference in Closing Price', fontsize=16, fontdict=dict(weight='bold'))
