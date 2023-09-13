import pandas as pd
import yfinance as yf
import numpy as np
import finance as fin
import warnings

def data_collection():
    
    dataset=pd.read_csv('ETFs.csv')
    symbols=fin.get_symbols(dataset)

    prices=fin.get_dataset(symbols)
    returns=prices.pct_change().dropna()
    
    return returns
    

def create_pf(strategy):

    returns=data_collection()
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

def pf_results(weights,returns):

    ann_ret=fin.annualize_rets(returns,len(returns))
    risk_free_rate=fin.rfr()

    pf_ret=fin.portfolio_return(weights,ann_ret)*100
    pf_vol=fin.portfolio_vol(weights,fin.cc_cov(returns))*100
    pf_sr=(pf_ret-risk_free_rate)/pf_vol
    pf_dd=max(np.sum(weights*returns,axis=1).cummax())*100

    portfolio={
        'Portfolio Returns':[pf_ret],
        'Portfolio Volatility':[pf_vol],
        'Sharpe Ratio': [pf_sr],
        'Max drawdown': [pf_dd]
    }

    return portfolio
    #return pd.DataFrame.from_dict(portfolio)

