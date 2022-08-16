"""
Contains functions that are commonly used throughout the app's code
"""

import numpy as np
import pandas as pd
import datetime

def simple_returns(prices):
    """
    Compute simple returns from a timeseries of prices.

    :param prices: (pd.Series, pd.DataFrame, np.ndarray) Prices of assets in wide-format,
                                                         with assets as columns, and indexed by 
                                                         datetimes.
    :return: (pd.Series, pd.DataFrame, np.ndarray) Returns of assets in wide-format, with assets 
                                                   as columns, and indexed by datetimes.
    .. notes::  Obtained from https://github.com/quantopian/empyrical  
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = prices.pct_change()
        # Add leading zero instead of NaN
        out.iloc[0,:] = 0
    else:
        # Assume np.ndarray
        out = np.diff(prices, axis=0)
        np.divide(out, prices[:-1], out=out)

    return out

def cum_returns(returns, starting_value=1, out=None):
    """
    Compute cumulative returns from simple returns.

    :param returns: (pd.Series, pd.DataFrame, np.ndarray) Returns of the strategy as a percentage, 
                                                          noncumulative.
    :param starting_value: (float) The starting returns. Optional.
    :param out: (pd.Series, pd.DataFrame, np.ndarray) Array to use as output buffer.
                                                      If not passed, a new array will be created.
    :return: (pd.Series, pd.DataFrame, np.ndarray) Series of cumulative returns,
                                                   indexed by datetimes.
    .. notes::  Obtained from https://github.com/quantopian/empyrical  
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out

def log_returns(prices):
    """
    Compute logarithmic returns from a timeseries of prices.

    :param prices: (pd.Series, pd.DataFrame, np.ndarray) Prices of assets in wide-format,
                                                         with assets as columns, and indexed by 
                                                         datetimes.
    :return: (pd.Series, pd.DataFrame, np.ndarray) Returns of assets in wide-format, with assets 
                                                   as columns, and indexed by datetimes.
    .. notes::  Obtained from https://github.com/quantopian/empyrical  
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = np.log(prices/prices.shift(1))#prices.pct_change()
        # out = np.log(prices) - np.log(prices.shift(1))
        
        # Add leading zero instead of NaN
        out.iloc[0,:] = 0
    else:
        # Assume np.ndarray
        out = np.diff(prices, axis=0)
        np.divide(out, prices[:-1], out=out)

    return out
    
def cum_log_returns(log_returns, starting_value=0, out=None):
    """
    Compute cumulative returns from logarithmic returns.    

    Parameters
    ----------
    log_returns : pd.Series, np.ndarray, or pd.DataFrame
        Logarithmic returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::
            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
         - Also accepts two dimensional data. In this case, each column is
           cumulated.
    starting_value : float, optional
       The starting returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
        
    Source
    ------
    https://github.com/quantopian/empyrical
    """
    if len(log_returns) < 1:
        return log_returns.copy()
    
    # Set nan values to 0
    nanmask = np.isnan(log_returns)
    if np.any(log_returns):
        log_returns = log_returns.copy()
        log_returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(log_returns)

    # Cumulative logarithmic returns    
    np.exp(log_returns.cumsum(), out = out)
    #np.subtract(out,1,out = out)
    
    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if log_returns.ndim == 1 and isinstance(log_returns, pd.Series):
            out = pd.Series(out, index=log_returns.index)
        elif isinstance(log_returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=log_returns.index, columns=log_returns.columns,
            )

    return out