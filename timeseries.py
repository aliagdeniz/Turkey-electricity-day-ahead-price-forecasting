import seaborn as sns
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd


def plot_average_prices(data):
    
    """Plot monthly, daily and hourly averages of input data."""
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), sharey=False)
    aylar = data.groupby(data.ay).mean()
    aylar.ptf.plot(ax=ax1, kind='bar')
    ax1.set_title('Monthly Average', fontsize=14)
    ax1.set_ylabel('Day-Ahead price in TL', fontsize=14)
    ax1.set_xlabel('Month of the year', fontsize=14)
    ax1.set_xticklabels(aylar.index.values, rotation=90)

    gunler = data.groupby(data.gun).mean()
    gunler.ptf.plot(ax=ax2, kind='bar')
    ax2.set_title('Weekly Average', fontsize=14)
    ax2.set_ylabel('Day-Ahead price in TL', fontsize=14)
    ax2.set_xlabel('Day of the week', fontsize=14)
    ax2.set_xticklabels(gunler.index.values, rotation=90)

    saatler = data.groupby(data.saat).mean()
    saatler.ptf.plot(ax=ax3, kind='bar')
    ax2.set_title('Hourly Average', fontsize=14)
    ax2.set_ylabel('Day-Ahead price in TL', fontsize=14)
    ax2.set_xlabel('Hour of the day', fontsize=14)
    ax3.set_xticklabels(saatler.index.values, rotation=90)
    
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)
    
    
def replace_outliers(data, column, tolerance):

    """Replace outliers out of 75% + tolerance * IQR or 25% - tolerance * IQR by these thresholds"""
    
    tol = tolerance
    data_prep = data.copy(deep=True)
    
    # calculate quantiles and inter-quantile range of the data
    q75 = data_prep[column].quantile(.75)
    q25 = data_prep[column].quantile(.25)
    IQR = q75 - q25

    # values larger (smaller) than q75 (q25) plus 'tol' times IQR get replaced by that value
    data_prep[column] = data_prep[column].apply(lambda x: q75 + tol * IQR if (x > q75 + tol * IQR) else x)
    data_prep[column] = data_prep[column].apply(lambda x: q25 - tol * IQR if (x < q75 - tol * IQR) else x)
    
    return data_prep

