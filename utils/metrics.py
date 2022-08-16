"""
Contains functions to obtain risk and performance metrics
"""

import numpy as np
import pandas as pd
import datetime

from .utils import (
    simple_returns,
    cum_returns,
    log_returns,
    cum_log_returns,
)

def get_summary_metrics(df_prices_1, df_prices_2):

    # Average returns
    avg_ret_1 = simple_returns(df_prices_1).mean()[0]*252
    avg_ret_2 = simple_returns(df_prices_2).mean()[0]*252

    # Volatility
    vol_1 = simple_returns(df_prices_1).std()[0]*np.sqrt(252)
    vol_2 = simple_returns(df_prices_2).std()[0]*np.sqrt(252)

    # Sharpe Ratio
    sr_1 = avg_ret_1 / vol_1
    sr_2 = avg_ret_2 / vol_2

    # Max Drawdown
    dd_1 = -100*np.min(get_historical_drawdown(df_prices_1))[0]
    dd_2 = -100*np.min(get_historical_drawdown(df_prices_2))[0]

    # VaR 99%
    var_1 = value_at_risk(simple_returns(df_prices_1), cutoff=0.01)
    var_2 = value_at_risk(simple_returns(df_prices_2), cutoff=0.01)

    # CVaR 99%
    cvar_1 = conditional_value_at_risk(simple_returns(df_prices_1), cutoff=0.01)
    cvar_2 = conditional_value_at_risk(simple_returns(df_prices_2), cutoff=0.01)


    df_summary_metrics = pd.DataFrame({
            df_prices_1.columns[0]:[
                avg_ret_1,vol_1,sr_1,dd_1,var_1,cvar_1,
            ],
            df_prices_2.columns[0]:[
                avg_ret_2,vol_2,sr_2,dd_2,var_2,cvar_2,
            ],
        }
        ,
        # index = ['Avg. Returns', 'Volatility','Sharpe Ratio', 'Max Drawdown','VaR 99%','CVaR 99%']
        index = ['Avg. Returns', 'Volatility','Sharpe Ratio', 'Max Drawdown','VaR 99%','CVaR 99%']
    )
    df_summary_metrics.index.name = 'Metrics'

    # Reorder index
    # df_summary_metrics = df_summary_metrics.loc[['VaR 99%', 'Volatility', 'Avg. Returns',  'CVaR 99%', 'Sharpe Ratio', 'Max Drawdown',],:]

    return df_summary_metrics

def get_historical_drawdown(asset_prices):
    """
    Obtain the drawdown of a product.

    :param asset_prices: (pd.Series, pd.DataFrame, np.ndarray) Prices of the product, indexed by 
                                                               datetimes.
    :return: (pd.Series, pd.DataFrame, np.ndarray) Drawdown values of the product.
    """
    running_max = np.maximum.accumulate(asset_prices)
    # running_max[running_max &lt; 1] = running_max
    drawdown = (asset_prices)/running_max - 1

    return drawdown



def value_at_risk(returns, cutoff=0.01):
    """
    Value at risk (VaR) of a returns stream.

    :param returns: (pd.Series or np.ndarray)  Daily returns of the product, noncumulative.
    :param cutoff: (float) Decimal representing the percentage cutoff for the bottom percentile of
                           returns. Defaults to 0.05.
    :return: (float) The VaR value       
    .. notes::  Obtained from https://github.com/quantopian/empyrical
    """
    return np.percentile(returns, 100 * cutoff)

def conditional_value_at_risk(returns, cutoff=0.01):
    """
    Conditional value at risk (CVaR) of a returns stream.

    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    ``cutoff`` as a percentile of all daily returns.

    :param returns: (pd.Series or np.ndarray)  Daily returns of the product, noncumulative.
    :param cutoff: (float) Decimal representing the percentage cutoff for the bottom percentile of
                           returns. Defaults to 0.05.
    :return: (float) The CVaR value       
    .. notes::  Obtained from https://github.com/quantopian/empyrical
    """
    # PERF: Instead of using the 'value_at_risk' function to find the cutoff
    # value, which requires a call to numpy.percentile, determine the cutoff
    # index manually and partition out the lowest returns values. The value at
    # the cutoff index should be included in the partition.
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index, axis = 0)[:cutoff_index + 1])
