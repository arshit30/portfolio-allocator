import pandas as pd
import yfinance as yf
import numpy as np
import finance as fin
import warnings

dataset=pd.read_csv('ETFs.csv')
symbols=fin.get_symbols(dataset)

prices=fin.get_dataset(symbols)
returns=prices.pct_change().dropna()
ann_ret=fin.annualize_rets(returns,len(returns))
risk_free_rate=fin.rfr()

def create_pf(returns,strategy):

    ann_ret=fin.annualize_rets(returns,len(returns))

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

while True:

    sts=input('Enter :')
    weights=create_pf(returns,sts)
    pf_ret=fin.portfolio_return(weights,ann_ret)
    pf_vol=fin.portfolio_vol(weights,fin.cc_cov(returns))
    pf_sr=(pf_ret-risk_free_rate)/pf_vol
    pf_dd=max(np.sum(weights*returns,axis=1).cummax())
    print('Return:',pf_ret)
    print('Vol:',pf_vol)
    print('Sharpe ratio:',pf_sr)
    print('Drawdown:',pf_dd)

