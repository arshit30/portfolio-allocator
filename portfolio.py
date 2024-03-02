import pandas as pd
import yfinance as yf
import numpy as np
import finance as fin
import warnings
import unittest
from sqlalchemy import create_engine
import pymysql
from sqlite3 import connect

def data_collection():
    
    dataset=pd.read_csv('ETFs.csv')
    symbols=fin.get_symbols(dataset)

    prices=fin.get_dataset(symbols)
    returns=prices.pct_change().dropna()
    
    engine = create_engine("mysql://ln90fus9zps3kf5c:nm089pvb9w9821bx@uyu7j8yohcwo35j3.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/mr76786mt2aisvgx")

    db=engine.connect()

    try:
        returns.to_sql(con=engine,name='Returns', if_exists='replace',index=False);
    except ValueError as vx:
        print(vx)
    except Exception as ex:
        print(ex)
    else:
        print('Data written to database')
    finally:
        db.close()  

def read_prices():
    
    engine = create_engine("mysql://ln90fus9zps3kf5c:nm089pvb9w9821bx@uyu7j8yohcwo35j3.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/mr76786mt2aisvgx")
    db=engine.connect()

    try:
        returns=pd.read_sql_table(table_name="Returns", con=db)
    except ValueError as vx:
        print(vx)
    except Exception as ex:
        print(ex)
    else:
        print('Returns read from database')
    finally:
        db.close()
    
    return returns

def create_pf(strategy):

    data_collection()

    returns=read_prices()
    ann_ret=fin.annualize_rets(returns,len(returns))
    risk_free_rate=fin.rfr()

    if strategy=='EW':
        weights=np.repeat(1/len(ann_ret),len(ann_ret))
    elif strategy=='MSR':
        weights=fin.msr(0.287,ann_ret,fin.cc_cov(returns))
    elif strategy=='GMV':
        weights=fin.gmv(fin.cc_cov(returns))
    elif strategy=='ERW':
        weights=fin.weight_erc(returns)
    
    weights=np.round(weights,4)
    return weights

def pf_results(weights):

    returns=read_prices()
    ann_ret=fin.annualize_rets(returns,len(returns))
    risk_free_rate=fin.rfr()

    pf_ret=fin.portfolio_return(weights,ann_ret)*100
    pf_vol=fin.portfolio_vol(weights,fin.cc_cov(returns))*100
    pf_sr=(pf_ret-risk_free_rate)/pf_vol
    pf_dd=max(np.sum(weights*returns,axis=1).cummax())*100

    portfolio={
        'Portfolio Returns':[str(np.round(pf_ret,2))+"%"],
        'Portfolio Volatility':[str(np.round(pf_vol,2))+"%"],
        'Sharpe Ratio': [str(np.round(pf_sr,2))+"%"],
        'Max drawdown': [str(np.round(pf_dd,2))+"%"]
    }

    dataset=pd.read_csv('ETFs.csv')
    portfolios=pd.DataFrame()
    portfolios['Index']=dataset['Index']
    portfolios['Weights']=weights
    portfolios.set_index(['Index'],inplace=True)
    
    portfolios=portfolios.to_dict()
    
    pf = {**portfolio, **portfolios}
    
    return pf

    #return pd.DataFrame.from_dict(portfolio)

