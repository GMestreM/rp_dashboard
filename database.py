from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
from dotenv import load_dotenv

import pandas as pd 
import numpy as np 

from utils.metrics import (
    get_historical_drawdown,
    get_summary_metrics,
)
from utils.utils import (
    simple_returns,
    cum_returns,
)

# Set global variables
LAG_PERF = 1

# Get env variables
load_dotenv()

DB_USER_NAME = os.environ.get('DB_USER_NAME', 'Unable to retrieve DB_USER_NAME') # environ.get
DB_USER_PWD  = os.environ.get('DB_USER_PWD', 'Unable to retrieve DB_USER_PWD')
DB_URL_PATH  = os.environ.get('DB_URL_PATH', 'Unable to retrieve DB_URL_PATH')
DB_URL_PORT  = os.environ.get('DB_URL_PORT', 'Unable to retrieve DB_URL_PORT')

# Create engine
engine = create_engine(f'postgresql://{DB_USER_NAME}:{DB_USER_PWD}@{DB_URL_PATH}:{DB_URL_PORT}/postgres')
engine.connect()

Base = declarative_base()

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

def query_recent_asset_data():
    """Retrieve the most recent asset price for each date
    
    This function performs an SQL query to retrive
    the most recent asset data for each date: if there
    is more than one value for the day, this function
    obtains those values with a closer 'timestamp' value.

    Returns:
        pd.DataFrame: A dataframe containing one row for
        each id_asset and date, filtered by its maximum timestamp.
    """
    from models import AssetPrices
    from sqlalchemy import func, and_
    import pandas as pd

    sql_subquery = session.query(
        AssetPrices.id_asset,
        AssetPrices.date,
        func.max(AssetPrices.timestamp).label('maxtimestamp')
    ).group_by(AssetPrices.id_asset, AssetPrices.date).subquery('t2')


    sql_query = session.query(AssetPrices).join(
        sql_subquery,
        and_(
            AssetPrices.id_asset == sql_subquery.c.id_asset,
            AssetPrices.date == sql_subquery.c.date,
            AssetPrices.timestamp == sql_subquery.c.maxtimestamp,
        )
    ).order_by(AssetPrices.id_asset, AssetPrices.date)

    df_recent_prices = pd.read_sql(sql_query.statement,session.bind)
    df_recent_prices['date'] = pd.to_datetime(df_recent_prices['date'], format = '%Y-%m-%d')

    return df_recent_prices

def query_recent_model_weights():
    """Retrieve the most recent model weights for each date and model
    
    This function performs an SQL query to retrive
    the most recent model weights for each date and model: if there
    is more than one value for the day, this function
    obtains those values with a closer 'timestamp' value.

    Returns:
        pd.DataFrame: A dataframe containing one row for
        each id_asset, id_model and date, filtered by its maximum timestamp.
    """
    from models import ModelWeights
    from sqlalchemy import func, and_
    import pandas as pd

    sql_subquery = session.query(
        ModelWeights.id_model,
        ModelWeights.id_asset,
        ModelWeights.date,
        func.max(ModelWeights.timestamp).label('maxtimestamp')
    ).group_by(ModelWeights.id_model, ModelWeights.id_asset, ModelWeights.date).subquery('t2')


    sql_query = session.query(ModelWeights).join(
        sql_subquery,
        and_(
            ModelWeights.id_model == sql_subquery.c.id_model,
            ModelWeights.id_asset == sql_subquery.c.id_asset,
            ModelWeights.date == sql_subquery.c.date,
            ModelWeights.timestamp == sql_subquery.c.maxtimestamp,
        )
    ).order_by(ModelWeights.date, ModelWeights.id_asset)

    df_recent_weights = pd.read_sql(sql_query.statement,session.bind)
    df_recent_weights['date'] = pd.to_datetime(df_recent_weights['date'], format = '%Y-%m-%d')

    return df_recent_weights

def get_asset_header():
    """Retrieve the headers of each asset in the database

    Returns:
        pd.DataFrame: Dataframe containing the detailed information
        of each asset stored in the database.
    """
    from models import AssetHeader
    
    df_sql_assets = pd.read_sql(session.query(AssetHeader).statement,session.bind)
    
    return df_sql_assets
    
def get_active_model_header():
    """Retrieve the details of the active model in the database

    Returns:
        pd.DataFrame: Dataframe containing the detailed information
        of the active model in the database.
    """
    from models import ModelHeader
    
    df_sql_models = pd.read_sql(session.query(ModelHeader).filter(ModelHeader.flag == True).statement,session.bind)
    
    return df_sql_models

def get_strategy_prices():
    """Retrieve cumulative returns of the active strategy and its benchmark

    Returns:
        dict: Dictionary where each field contains a pd.DataFrame with
        relevant information about the active strategy.
    """
        
    # Obtain active model
    model_row = get_active_model_header()
    
    # Parse model configuration
    asset_list_model = model_row['assets'].values[0].split(';')
    id_model         = model_row['id']
    window_size      = model_row['window_size'].values[0]
    
    # Obtain asset price data
    df_recent_asset_prices = query_recent_asset_data()
    
    # Transform to tidy format using close prices
    df_asset_prices_model = pd.DataFrame()
    for asset_id in asset_list_model:
        # Filter
        df_asset_prices_filt = df_recent_asset_prices.loc[df_recent_asset_prices['id_asset'] == int(asset_id),['close_price','date']].copy()
        df_asset_prices_filt.set_index(['date'], inplace=True)
        df_asset_prices_filt.columns = [asset_id]
        
        # Join
        df_asset_prices_model = pd.concat([df_asset_prices_model, df_asset_prices_filt], axis = 1)
    
    # Get returns
    df_asset_return_model = simple_returns(df_asset_prices_model)
    
    # Obtain model weights
    df_rp_allocation_backtest = query_recent_model_weights()
    
    # Filter model weights for the active model
    df_rp_allocation_backtest_active = df_rp_allocation_backtest.loc[df_rp_allocation_backtest['id_model'] == id_model.values[0],:].copy()
    
    # Transform weights to tidy format
    df_model_weights = pd.DataFrame()
    for asset_id in asset_list_model:
        # Filter
        df_rp_allocation_backtest_active_filt = df_rp_allocation_backtest_active.loc[df_rp_allocation_backtest_active['id_asset'] == int(asset_id),['weight','date']].copy()
        df_rp_allocation_backtest_active_filt.set_index(['date'], inplace=True)
        df_rp_allocation_backtest_active_filt.columns = [asset_id]
        
        # Join
        df_model_weights = pd.concat([df_model_weights, df_rp_allocation_backtest_active_filt], axis = 1)
        
    # Get RP model performance 
    df_ret_rp = (df_model_weights.shift(LAG_PERF) * df_asset_return_model).sum(axis = 1).fillna(0)
    
    # Get Equally-weighted performance
    df_ew_allocation_backtest = df_model_weights.copy()*0
    df_ew_allocation_backtest = df_ew_allocation_backtest + 1/len(asset_list_model)
    df_ret_ew = (df_ew_allocation_backtest.shift(LAG_PERF) * df_asset_return_model).sum(axis = 1).fillna(0)
    
    df_ret_rp = df_ret_rp.iloc[window_size:]
    df_ret_ew = df_ret_ew.iloc[window_size:]
    
    # Return data starting from 2014
    df_ret_rp = df_ret_rp.loc[df_ret_rp.index >= pd.to_datetime('2014-01-01',format = '%Y-%m-%d')]
    df_ret_ew = df_ret_ew.loc[df_ret_ew.index >= pd.to_datetime('2014-01-01',format = '%Y-%m-%d')]

    df_perf_rp = cum_returns(df_ret_rp,starting_value = 100)
    df_perf_ew = cum_returns(df_ret_ew,starting_value = 100)
    
    # Prepare output dictionary with relevant dataframes
    dict_output = {
        'df_perf_rp'            :df_perf_rp,
        'df_perf_ew'            :df_perf_ew,
        'df_model_weights'      :df_model_weights,
        'df_asset_return_model' :df_asset_return_model,
    }
    return dict_output
   