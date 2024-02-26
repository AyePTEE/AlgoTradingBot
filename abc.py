import pandas as pd 
import matplotlib.pyplot as plt
import requests
import numpy as np
import streamlit as st
from math import floor
from termcolor import colored as cl 
import streamlit as st
import yfinance as yf 
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)
title = st.text_input('stock code')
d=st.date_input("start date")
ibm = yf.download(title,d)
st.title("Dataset")
st.dataframe(ibm)
st.set_option('deprecation.showPyplotGlobalUse', False)
def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    return plus_di, minus_di, adx_smooth

ibm['plus_di'] = pd.DataFrame(get_adx(ibm['High'], ibm['Low'], ibm['Close'], 14)[0]).rename(columns = {0:'plus_di'})
ibm['minus_di'] = pd.DataFrame(get_adx(ibm['High'], ibm['Low'], ibm['Close'], 14)[1]).rename(columns = {0:'minus_di'})
ibm['adx'] = pd.DataFrame(get_adx(ibm['High'], ibm['Low'], ibm['Close'], 14)[2]).rename(columns = {0:'adx'})
ibm = ibm.dropna()


def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    
    return rsi_df[3:]

ibm['rsi_14'] =get_rsi(ibm['Close'], 14)
ibm = ibm.dropna()
ibm.tail()

def sma(data, lookback):
    sma = data.rolling(lookback).mean()
    return sma

def get_bb(data, lookback):
    std = data.rolling(lookback).std()
    upper_bb = sma(data, lookback) + std * 2
    lower_bb = sma(data, lookback) - std * 2
    middle_bb = sma(data, lookback)
    return upper_bb, middle_bb, lower_bb

ibm['upper_bb'], ibm['middle_bb'], ibm['lower_bb'] = get_bb(ibm['Close'], 20)
ibm.tail()

def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(alpha = 1/atr_lookback).mean()
    
    kc_middle = close.ewm(kc_lookback).mean()
    kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
    kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
    
    return kc_middle, kc_upper, kc_lower
    
ibm['kc_middle'], ibm['kc_upper'], ibm['kc_lower'] = get_kc(ibm['High'], ibm['Low'], ibm['Close'], 20, 2, 10)
ibm.tail()
st.title("Updated Dataset")
st.dataframe(ibm)
def bb_kc_rsi_strategy(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi):
    buy_price = []
    sell_price = []
    bb_kc_rsi_signal = []
    signal = 0
    
    for i in range(len(prices)):
        if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < 30:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
                
        elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > 70:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_kc_rsi_signal.append(0)
                        
    return buy_price, sell_price, bb_kc_rsi_signal

buy_price, sell_price, bb_kc_rsi_signal = bb_kc_rsi_strategy(ibm['Close'], ibm['upper_bb'], ibm['lower_bb'], ibm['kc_upper'], ibm['kc_lower'], ibm['rsi_14'])

position = []
for i in range(len(bb_kc_rsi_signal)):
    if bb_kc_rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(ibm['Close'])):
    if bb_kc_rsi_signal[i] == 1:
        position[i] = 1
    elif bb_kc_rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
kc_upper = ibm['kc_upper']
kc_lower = ibm['kc_lower']
upper_bb = ibm['upper_bb'] 
lower_bb = ibm['lower_bb']
rsi = ibm['rsi_14']
close_price = ibm['Close']
bb_kc_rsi_signal = pd.DataFrame(bb_kc_rsi_signal).rename(columns = {0:'bb_kc_rsi_signal'}).set_index(ibm.index)
position = pd.DataFrame(position).rename(columns = {0:'bb_kc_rsi_position'}).set_index(ibm.index)

frames = [close_price, kc_upper, kc_lower, upper_bb, lower_bb, rsi, bb_kc_rsi_signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)

strategy.tail()

ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
ax1.plot(ibm['Close'], linewidth = 2.5)
ax1.set_title('{} CLOSE PRICE'.format(title))
ax2.plot(ibm['rsi_14'], color = 'orange', linewidth = 2.5)
ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
ax2.set_title('{} RELATIVE STRENGTH INDEX'.format(title))
a=plt.show()
st.title("RSI Plotting")
st.pyplot(a)

ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
ax1.plot(ibm['Close'], linewidth = 2, color = '#ff9800')
ax1.set_title('{} CLOSING PRICE'.format(title))
ax2.plot(ibm['plus_di'], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
ax2.plot(ibm['minus_di'], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
ax2.plot(ibm['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
ax2.legend()
ax2.set_title('{} ADX '.format(title))
st.title("ADX Plotting")
b= plt.show()
st.pyplot(b)

ibm['Close'].plot(label = 'CLOSE PRICES', color = 'skyblue')
ibm['upper_bb'].plot(label = 'UPPER BB 20', linestyle = '--', linewidth = 1, color = 'black')
ibm['middle_bb'].plot(label = 'MIDDLE BB 20', linestyle = '--', linewidth = 1.2, color = 'grey')
ibm['lower_bb'].plot(label = 'LOWER BB 20', linestyle = '--', linewidth = 1, color = 'black')
plt.legend(loc = 'upper left')
plt.title('{} BOLLINGER BANDS'.format(title))
st.title("BOLLINGER BANDS Plotting")
c= plt.show()
st.pyplot(c)

plt.plot(ibm['Close'], linewidth = 2, label = 'ibm')
plt.plot(ibm['kc_upper'], linewidth = 2, color = 'orange', linestyle = '--', label = 'KC UPPER 20')
plt.plot(ibm['kc_middle'], linewidth = 1.5, color = 'grey', label = 'KC MIDDLE 20')
plt.plot(ibm['kc_lower'], linewidth = 2, color = 'orange', linestyle = '--', label = 'KC LOWER 20')
plt.legend(loc = 'lower right')
plt.title('{} KELTNER CHANNEL'.format(title))
st.title("KELTNER CHANNEL Plotting")
d=plt.show()
st.pyplot(d)
ibm_ret = pd.DataFrame(np.diff(ibm['Close'])).rename(columns = {0:'returns'})
bb_kc_rsi_strategy_ret = []

for i in range(len(ibm_ret)):
    returns = ibm_ret['returns'][i]*strategy['bb_kc_rsi_position'][i]
    bb_kc_rsi_strategy_ret.append(returns)
    
bb_kc_rsi_strategy_ret_df = pd.DataFrame(bb_kc_rsi_strategy_ret).rename(columns = {0:'bb_kc_rsi_returns'})
investment_value = 100000
number_of_stocks = floor(investment_value/ibm['Close'][0])
bb_kc_rsi_investment_ret = []

for i in range(len(bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'])):
    returns = number_of_stocks*bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'][i]
    bb_kc_rsi_investment_ret.append(returns)

bb_kc_rsi_investment_ret_df = pd.DataFrame(bb_kc_rsi_investment_ret).rename(columns = {0:'investment_returns'})
total_investment_ret = round(sum(bb_kc_rsi_investment_ret_df['investment_returns']), 2)
profit_percentage = ((total_investment_ret/investment_value)*100)
st.title("Result by BB KC RSI strategy")
st.subheader('Profit gained from the BB KC RSI strategy by investing 100k in ibm : {}'.format(total_investment_ret))
st.subheader('Profit percentage of the BB KC RSI strategy : {}%'.format(profit_percentage))

def adx_rsi_strategy(prices, adx, pdi, ndi, rsi):
    buy_price1 = []
    sell_price1 = []
    adx_rsi_signal = []
    signal = 0
    
    for i in range(len(prices)):
        if adx[i] > 35 and pdi[i] < ndi[i] and rsi[i] < 50:
            if signal != 1:
                buy_price1.append(prices[i])
                sell_price1.append(np.nan)
                signal = 1
                adx_rsi_signal.append(signal)
            else:
                buy_price1.append(np.nan)
                sell_price1.append(np.nan)
                adx_rsi_signal.append(0)
                
        elif adx[i] > 35 and pdi[i] > ndi[i] and rsi[i] > 50:
            if signal != -1:
                buy_price1.append(np.nan)
                sell_price1.append(prices[i])
                signal = -1
                adx_rsi_signal.append(signal)
            else:
                buy_price1.append(np.nan)
                sell_price1.append(np.nan)
                adx_rsi_signal.append(0)
        else:
            buy_price1.append(np.nan)
            sell_price1.append(np.nan)
            adx_rsi_signal.append(0)
                        
    return buy_price1, sell_price1, adx_rsi_signal

buy_price1, sell_price1, adx_rsi_signal = adx_rsi_strategy(ibm['Close'], ibm['adx'], ibm['plus_di'], ibm['minus_di'], ibm['rsi_14'])


position1 = []
for i in range(len(adx_rsi_signal)):
    if adx_rsi_signal[i] > 1:
        position1.append(0)
    else:
        position1.append(1)
        
for i in range(len(ibm['Close'])):
    if adx_rsi_signal[i] == 1:
        position1[i] = 1
    elif adx_rsi_signal[i] == -1:
        position1[i] = 0
    else:
        position1[i] = position1[i-1]
        
adx = ibm['adx']
pdi = ibm['plus_di']
ndi = ibm['minus_di']
rsi = ibm['rsi_14'] 
close_price = ibm['Close']
adx_rsi_signal = pd.DataFrame(adx_rsi_signal).rename(columns = {0:'adx_rsi_signal'}).set_index(ibm.index)
position1 = pd.DataFrame(position1).rename(columns = {0:'adx_rsi_position1'}).set_index(ibm.index)

frames = [close_price, adx, pdi, ndi, rsi, adx_rsi_signal, position1]
strategy1 = pd.concat(frames, join = 'inner', axis = 1)


print(strategy1)


ibm_ret = pd.DataFrame(np.diff(ibm['Close'])).rename(columns = {0:'returns'})
adx_rsi_strategy_ret = []

for i in range(len(ibm_ret)):
    returns = ibm_ret['returns'][i]*strategy1['adx_rsi_position1'][i]
    adx_rsi_strategy_ret.append(returns)
    
adx_rsi_strategy_ret_df = pd.DataFrame(adx_rsi_strategy_ret).rename(columns = {0:'adx_rsi_returns'})
investment_value = 100000
number_of_stocks1 = floor(investment_value/ibm['Close'][0])
adx_rsi_investment_ret = []

for i in range(len(adx_rsi_strategy_ret_df['adx_rsi_returns'])):
    returns = number_of_stocks*adx_rsi_strategy_ret_df['adx_rsi_returns'][i]
    adx_rsi_investment_ret.append(returns)

adx_rsi_investment_ret_df = pd.DataFrame(adx_rsi_investment_ret).rename(columns = {0:'investment_returns1'})
total_investment_ret1 = round(sum(adx_rsi_investment_ret_df['investment_returns1']), 2)
profit_percentage1 = ((total_investment_ret1/investment_value)*100)
st.title("Result by ADX RSI strategy")
st.subheader('Profit gained from the ADX RSI strategy by investing 100k in ibm : {}'.format(total_investment_ret1))
st.subheader('Profit percentage of the ADX RSI strategy : {}%'.format(profit_percentage1))

st.title("Final Result")

if(total_investment_ret1>total_investment_ret):
    st.subheader("ADX RSI is a better a strategy".format())
    st.subheader("There will be a profit of {} if we use ADX RSI".format(total_investment_ret1-total_investment_ret))
    st.subheader("Profit Percentage will be {} if we use ADX RSI".format(profit_percentage1-profit_percentage))
elif(total_investment_ret1<total_investment_ret):
    st.title("BB KC RSI is a better a strategy")
    st.subheader("There will be a profit of {} if we use BB KC RSI".format(total_investment_ret-total_investment_ret1))
    st.subheader("Profit Percentage will be {}% if we use BB KC RSI".format(profit_percentage-profit_percentage1))