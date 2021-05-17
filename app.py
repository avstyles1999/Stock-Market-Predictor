import yfinance
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from func import calc_RSI, calc_avg_Volume, calc_MACD, calc_MFI, calc_AO, uptrend, downtrend, stock_prediction, view_data
from datetime import date

from flask import Flask, render_template, url_for

PEOPLE_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

li = []
mi = []

#stock_prediction('SBIN.NS','SBI')
dataset = pd.read_csv('nifty50.csv')
#print(dataset)
for i in range(50):
    #print(i)
    stock_prediction(dataset['Symbol'][i], dataset['Company Name'][i],li,mi)

#print(li)
#print(mi)


@app.route('/')
def hello_world():
    return render_template('index.html', uptrend_list = li, downtrend_list = mi)
    #return 'Hello, World!'

@app.route('/load/<string:name>/<string:symbol>')
def view(name,symbol):
    #print(symbol)
    view_data(symbol)
    time.sleep(2)
    return render_template('view.html',name=name,s=symbol)

if __name__ == "__main__":
    app.run(debug=True)