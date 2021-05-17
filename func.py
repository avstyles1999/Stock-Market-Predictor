import yfinance
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from matplotlib import pyplot as plt
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

def calc_RSI(diff,n=14):
    
    up,down = diff.copy(),diff.copy()
    
    up[up<0] = 0
    down[down>0] = 0
    
    #Calculate the exponential weighted values
    roll_up1 = up.ewm(com = n-1,min_periods = n,adjust = False,ignore_na = False).mean()
    roll_down1 = down.abs().ewm(com = n-1,min_periods = n,adjust = False,ignore_na = False).mean()
    
    RS = roll_up1/roll_down1
    
    #Calculate RSI
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    return RSI

def calc_avg_Volume(diff,n=14):
    
    short_rolling = diff.rolling(window=n).mean()
    
    return short_rolling

def calc_MACD(Close):

    #Calculate the Short Term Exponential Moving Average
    ShortEMA = Close.ewm(span=12, adjust=False).mean()

    #Calculate the Long Term Exponential Moving Average
    LongEMA = Close.ewm(span=26, adjust=False).mean()

    #Calculate MACD
    MACD = ShortEMA - LongEMA

    return MACD

def calc_MFI(data, n = 14):
    typical_price = (data['Close'] + data['High'] + data['Low'])/3

    money_flow = typical_price * data['Volume']
    
    #Get All Positive and negative Money Flows

    positive_flow =[] 
    negative_flow = []

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]: #if the present typical price is greater than yesterdays typical price
            positive_flow.append(money_flow[i-1])# Then append money flow at position i-1 to the positive flow list
            negative_flow.append(0) #Append 0 to the negative flow list
        elif typical_price[i] < typical_price[i-1]:#if the present typical price is less than yesterdays typical price
            negative_flow.append(money_flow[i-1])# Then append money flow at position i-1 to negative flow list
            positive_flow.append(0)#Append 0 to the positive flow list
        else: #Append 0 if the present typical price is equal to yesterdays typical price
            positive_flow.append(0)
            negative_flow.append(0)

    #Get All positive And Negative Money Flows Within The Time Period
    positive_mf =[]
    negative_mf = [] 

    #Get all of the positive money flows within the time period
    for i in range(n-1, len(positive_flow)):
        positive_mf.append(sum(positive_flow[i+1-n : i+1]))

    #Get all of the negative money flows within the time period  
    for i in range(n-1, len(negative_flow)):
        negative_mf.append(sum(negative_flow[i+1-n : i+1]))

    #Calculate MFI
    mfi = 100 * (np.array(positive_mf) / (np.array(positive_mf)  + np.array(negative_mf) ))

    return mfi

def calc_AO(data,fast=5,slow=34):
    
    sma1 = ((data['High'] + data['Low'])/2).rolling(window = fast).mean()
    sma2 = ((data['High'] + data['Low'])/2).rolling(window = slow).mean()

    #calculate AO
    ao = sma1 - sma2

    return ao
"""
def uptrend(price):

    up = 0
    down = 0
    trend = False
    
    for i in range(-7,-1):
        if price[price.index[i+1]] > price[price.index[i]]: 
            up += 1
            down = 0
            if up>=3:
                trend = True
        else:
            down +=  1
            up = 0
            if down >=3:
                trend = False    
    return trend
"""

def uptrend(price):

    trend = 0
    
    for i in range(-7,-1):
        if price[price.index[i]] == 1: 
            trend = 1
        elif price[price.index[i]] == 3:
            trend = 3   
    #l.append(trend)
    return trend

def downtrend(price):

    trend = 0
    
    for i in range(-7,-1):
        if price[price.index[i]] == 3: 
            trend = 3
        elif price[price.index[i]] == 1:
            trend = 1   
    #l.append(trend)
    return trend

def view_data(s):
    #plotting data

    if os.path.exists(f'./static/images/plot_{s}.png'):
        os.remove(f'./static/images/plot_{s}.png')

    data = yfinance.download(s,'1996-1-1',date.today())

    data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    data = data.dropna()
    data = data.reset_index()

    plt.figure(figsize=(12.2,4.5)) 
    plt.plot( data['Date'],data['Close'],  label='Close')
    plt.xlabel('Date',fontsize=14)
    plt.ylabel('Price USD ($)',fontsize=14)
    plt.savefig(f'./static/images/plot_{s}.png')

def stock_prediction(s,n,li,mi):
    #data = pd.read_csv("NSE SBIN - Sheet1.csv")
    #print(s)

    data = yfinance.download(s,'1996-1-1',date.today())
    #sbin = yfinance.Ticker("SBIN.NS")
    #data = sbin.history(period="max")
    data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    data = data.dropna()
    #data = clean_dataset(data)

    data = data.reset_index()

    

    for i in ['Open', 'High', 'Close', 'Low']: 
        data[i]  =  data[i].astype('float64')

    #print(data)

    data['Change'] = 0.0

    l = len(data)

    for i in range(1,l):
        data['Change'][i]=data['Close'][i]-data['Close'][i-1]
        
    #data['Change'][0] = 0


    rsi = calc_RSI(data['Change'])
    rsi[0:13] = 0.0 

    avg_Volume = calc_avg_Volume(data['Volume'])
    avg_Volume[0:13] = 0.0
    #print(rsi)

    data['RSI'] = rsi

    data['Avg Volume'] = avg_Volume

    macd = calc_MACD(data['Close'])

    signal = macd.ewm(span=9, adjust=False).mean()

    #print(macd)
    #print(data['RSI'][0:200])

    data['MACD'] = macd
    data['Signal Line'] = signal

    mfi = calc_MFI(data)
    for i in range(0,14):
        mfi = np.insert(mfi,0,0,axis=0)

    #print(len(mfi))
    #print(len(data))
    data['MFI'] = mfi

    AO = calc_AO(data)
    data['AO'] = AO

    #plotting data

    #plt.figure(figsize=(12.2,4.5)) 
    #plt.plot( data['Date'],data['Close'],  label='Close')
    #plt.xlabel('Date',fontsize=14)
    #plt.ylabel('Price USD ($)',fontsize=14)
    #plt.show()

    #plotting volume vs Average Volume

    #plt.xlabel('Date', fontsize=14)
    #plt.title('Avg Volume vs Volume')
    #plt.plot(data['Date'][0:200], data['Avg Volume'][0:200], label='Avg Volume', color='red')
    #plt.plot(data['Date'][0:200], data['Volume'][0:200], label='Volume', color='orange')
    #plt.legend(loc='upper left')
    #plt.show()

    #print(data['Avg Volume'][0:200])
    #plotting RSI

    #plt.xlabel('Date',fontsize = 14)
    #plt.ylabel('RSI',fontsize = 14)
    #plt.plot(data['Date'][0:200], data['RSI'][0:200])
    #plt.axhline(50, linestyle='--',color = 'red', label='Crossover Line')
    #plt.show()

    #plotting MACD and Signal Line

    #plt.xlabel('Date',fontsize = 14)
    #plt.ylabel
    #plt.plot(data['Date'][0:200], data['MACD'][0:200], label='AAPL MACD', color = 'red')
    #plt.plot(data['Date'][0:200], data['Signal Line'][0:200], label='Signal Line', color='blue')
    #plt.legend(loc='upper left')
    #plt.show()

    #plotting MFI

    #plt.plot( data['Date'][0:200], data['MFI'][0:200],  label='MFI')
    #plt.axhline(20, linestyle='--',color = 'orange', label='Over Sold Line (Buy)')  #Over Sold Line (Buy)
    #plt.axhline(80, linestyle='--', color = 'red', label='Over Bought Line (Sell)')  #Over Bought line (Sell)
    #plt.title('MFI')
    #plt.ylabel('MFI Values',fontsize=18)
    #plt.legend( loc='upper left')
    #plt.show()

    #plotting AO

    #plt.plot(data['Date'][0:200], data['AO'][0:200], label = 'AO')
    #plt.title('AO')
    #plt.ylabel('AO values',fontsize = 18)
    #plt.axhline(0, linestyle='--',color = 'red', label='Crossover Line') 
    #plt.legend(loc='upper left')
    #plt.show()

    data['Class'] = 5

    l = len(data['RSI'])

    initial_increase_volume_increase = []
    initial_increase_volume_decrease = []
    initial_decrease_volume_increase = []
    initial_decrease_volume_decrease = []


    for i in range(1,l):
        
        if data['Class'][i-1]==1:
            if(not(data['Change'][i]<=0 and abs(data['Change'][i]/data['Close'][i])*100>=1.5)):
                initial_increase_volume_increase.append(data['Date'][i])
                data['Class'][i]=1
                continue
        elif data['Class'][i-1]==3:
            if(not(data['Change'][i]>=0 and abs(data['Change'][i]/data['Close'][i])*100>=1.5)):
                initial_decrease_volume_decrease.append(data['Date'][i])
                data['Class'][i]=3
                continue
            
        if (data['RSI'][i-1] < 50 and data['RSI'][i] >= 50) and data['MFI'][i] >= 50 and (data['MACD'][i] > data['Signal Line'][i]) :
            if(i>=l-3):
                if data['Volume'][i] >= data['Avg Volume'][i]:
                    data['Class'][i] = 1
                else:
                    data['Class'][i] = 3
                continue
            x = str(data['Date'][i])
            y = str(data['Date'][i+3])
            year1 = x[0:4]
            moth1 = x[5:7]
            day1 = x[8:10]
            year2 = y[0:4]
            moth2 = y[5:7]
            day2 = y[8:10]
            date1 = date(int(year1),int(moth1),int(day1))
            date2 = date(int(year2),int(moth2),int(day2))
            delta = date2 - date1
            if (data['RSI'][i+3]-data['RSI'][i])/(delta.days) > 0:
                if data['Volume'][i] >= data['Avg Volume'][i]:
                    f = True
                    for j in range(i+1,i+3):
                        if data['Change'][j] > 0:
                            continue
                        else:
                            change = (data['Change'][j])/data['Close'][j]
                            if abs(change) * 100 < 1.5:
                                continue
                            f = False
                            break
                    if f:
                        initial_increase_volume_increase.append(data['Date'][i])
                        data['Class'][i] = 1
                    else:
                        initial_increase_volume_decrease.append(data['Date'][i])
                        data['Class'][i] = 3
        elif (data['RSI'][i-1] > 50 and data['RSI'][i] <= 50) and data['MFI'][i] <= 50 and (data['MACD'][i] < data['Signal Line'][i]) :
            if(i>=l-3):
                if data['Volume'][i] < data['Avg Volume'][i]:
                    data['Class'][i] = 3
                else:
                    data['Class'][i] = 1
                continue
            x = str(data['Date'][i])
            y = str(data['Date'][i+3])
            year1 = x[0:4]
            moth1 = x[5:7]
            day1 = x[8:10]
            year2 = y[0:4]
            moth2 = y[5:7]
            day2 = y[8:10]
            date1 = date(int(year1),int(moth1),int(day1))
            date2 = date(int(year2),int(moth2),int(day2))
            delta = date2 - date1
            if (data['RSI'][i+3]-data['RSI'][i])/(delta.days) < 0:
                if data['Volume'][i] < data['Avg Volume'][i]:
                    f = True
                    for j in range(i+1,i+3):
                        if data['Change'][j] < 0:
                            continue
                        else:
                            change = (data['Change'][j])/data['Close'][j]
                            if abs(change) * 100 < 1.5:
                                continue
                            f = False
                            break
                    if f:
                        initial_decrease_volume_decrease.append(data['Date'][i])
                        data['Class'][i] = 3
                    else:
                        initial_decrease_volume_increase.append(data['Date'][i])
                        data['Class'][i] = 1
        else:
            if data['Class'][i-1] == 5:
                if data['Close'][i]>data['Close'][i-1]:
                    data['Class'][i]=1
                else:
                    data['Class'][i]=3
            else:
                data['Class'][i]=data['Class'][i-1]
            


    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #print(len(initial_increase_volume_increase))
    #print(len(initial_increase_volume_decrease))
    #print(len(initial_decrease_volume_decrease))
    #print(len(initial_decrease_volume_increase))
    #print(data['Class'])
    #print(len(data))


    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Change', 'RSI', 'Avg Volume', 'MACD', 'Signal Line', 'MFI']]

    X = np.asarray(features)
    y = np.asarray(data['Class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

    #print(data['Class'][-20:])

    l = data['Class'].tail(7)
    
    if(uptrend(l) == 1):
        li.append([n,s])

    if(downtrend(l) == 3):
        mi.append([n,s])
    #print(np.any(np.isnan(X_train)))
    #print(np.any(np.isfinite(X_train)))
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    #print(classification_report(y_test, y_predict))


    df = pd.DataFrame(y_test, columns= ['a'])
    #print(df['a'].value_counts())