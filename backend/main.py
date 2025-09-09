from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import pandas as pd
import time
import signal
import sys
import threading
import requests
import random

import base64
import io
import matplotlib

matplotlib.use('Agg')  

app = Flask(__name__)
CORS(app)

activeRequests = {}
requestLock = threading.Lock()


session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

def getParameters(requestData):

    return {
        'sequenceLength': requestData.get('sequenceLength', 60),
        'futureDays': requestData.get('futureDays', 60),
        'trainSplit': requestData.get('trainSplit', 0.7),
        'nEstimators': requestData.get('nEstimators', 500),
        'maxDepth': requestData.get('maxDepth', 20),
        'maxFeatures': requestData.get('maxFeatures', 0.6),
        'minSamplesSplit': requestData.get('minSamplesSplit', 2),
        'minSamplesLeaf': requestData.get('minSamplesLeaf', 1)
    }

def getStockData(ticker_symbol, period="2y"):

    # imn trying everything to get it to stop rate limit hitting
    ticker = ticker_symbol.upper()
    
  
    def try_yfinance():
        try:
            stockObj = yf.Ticker(ticker, session=session)
       
            time.sleep(random.uniform(1, 3))
            
        
            data = stockObj.history(period=period, auto_adjust=True, timeout=30)
            
            if not data.empty:
                print(f"yfinance worked")
                return data
            
        except Exception as e:
            print(f"yfinance failed: {str(e)}")
            return None

    def try_yahoo_direct():
        try:
            time.sleep(random.uniform(2, 4))
            
            import datetime
            endTime = int(datetime.datetime.now().timestamp())
            startTime = int((datetime.datetime.now() - datetime.timedelta(days=730)).timestamp())
            
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"

            params = {
                'period1': startTime,
                'period2': endTime,
                'interval': '1d',
                'events': 'history',
                'includeAdjustedClose': 'true'
            }
            
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            data = pd.read_csv(StringIO(response.text), index_col='Date', parse_dates=True)
            
            if not data.empty:
                data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                data = data.drop('Adj Close', axis=1)  
                print(f"Yahoo direct worked and formatted as yfinancde")
                return data
                
        except Exception as e:
            print(f"Yahoo direct API failed: {str(e)}")
            return None

    def try_pandas_datareader():
        try:
            import pandas_datareader as pdr
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            time.sleep(random.uniform(1, 2))
            
            data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date, session=session)
            
            if not data.empty:
                print(f"Successfully retrieved {ticker} data via pandas-datareader")
                return data
                
        except Exception as e:
            print(f"pandas-datareader failed: {str(e)}")
            return None
    
    methods = [try_yfinance, try_yahoo_direct, try_pandas_datareader]
    
    for i, method in enumerate(methods, 1):
        try:
            data = method()
            if data is not None and not data.empty and len(data) > 120:
                return data
            else:
                print(f"Method {i} failed")
        except Exception as e:
            print(f"Method {i} failed with exception: {str(e)}")
    
    return None

def predictPrice(ticker_symbol, params=None):
    ticker = ticker_symbol.upper()

    if params is None:
        params = {
            'sequenceLength': 60,
            'futureDays': 60,
            'trainSplit': 0.7,
            'nEstimators': 500,
            'maxDepth': 20,
            'maxFeatures': 0.6,
            'minSamplesSplit': 2,
            'minSamplesLeaf': 1
        }
    
    length = params['sequenceLength']
    futureAmount = params['futureDays']
    
    try:
        print(f"Starting data retrieval for {ticker}")
        stock = getStockData(ticker)
        
        if stock is None or stock.empty:
            return {"error": f"Unable to retrieve data for ticker {ticker} from any source"}
            
        print(f"Retrieved {len(stock)} days of data for {ticker}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"Final error for {ticker}: {error_msg}")
        return {"error": f"Failed to retrieve data: {error_msg}"}
    
    if len(stock) < 120:
        return {"error": f"Insufficient data for {ticker}. Need at least 120 days of data n got {len(stock)} days"}
    
    y = stock['Close'].values

    stock['Price_Range'] = stock['High'] - stock['Low']
    stock['Volume_Price'] = stock['Volume'] * stock['Close']
    stock['adjCloseMA3'] = stock['Close'].rolling(window=3).mean()

    rfData = stock[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Range', 'Volume_Price', 'adjCloseMA3']].bfill().values

    def createSequences(data, length=60):
        X, y = [], []
        for i in range(length, len(data)):
            X.append(data[i-length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def createRfSequences(data, target, length=60):
        X, y = [], []
        for i in range(length, len(data)):
            X.append(data[i-length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    sequenceX, sequenceY = createSequences(y, length)
    rfSequenceX, rfSequenceY = createRfSequences(rfData, y, length)
    print(f"Sequences created: {len(sequenceX)} samples")

    if len(sequenceX) == 0:
        return {"error": f"Not enough data to create sequences for {ticker}. Need at least {length + 1} days of data."}
    
    if len(sequenceX) < 10:
        return {"error": f"Insufficient sequences for reliable prediction. Only {len(sequenceX)} sequences available, need at least 10."}

    # splitting to training /testing sections
    splitIndex = int(len(sequenceX) * params['trainSplit'])
    xTrain, xTest = sequenceX[:splitIndex], sequenceX[splitIndex:]
    yTrain, yTest = sequenceY[:splitIndex], sequenceY[splitIndex:]

    rfXTrain, rfXTest = rfSequenceX[:splitIndex], rfSequenceX[splitIndex:]
    rfYTrain, rfYTest = rfSequenceY[:splitIndex], rfSequenceY[splitIndex:]

    print(f"Training samples: {len(xTrain)}, Test samples: {len(xTest)}")

    xTrainFlat = xTrain.reshape(xTrain.shape[0], -1)
    xTestFlat = xTest.reshape(xTest.shape[0], -1)

    rfXTrainFlat = rfXTrain.reshape(rfXTrain.shape[0], -1)
    rfXTestFlat = rfXTest.reshape(rfXTest.shape[0], -1)

    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrainFlat)
    xTest = scaler.transform(xTestFlat)

    rfScaler = StandardScaler()
    rfXTrain = rfScaler.fit_transform(rfXTrainFlat)
    rfXTest = rfScaler.transform(rfXTestFlat)

    linearModel = LinearRegression()
    linearModel.fit(xTrainFlat, yTrain)

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [params['nEstimators']],
        "max_depth": [params['maxDepth'] if params['maxDepth'] > 0 else None],
        "max_features": [params['maxFeatures']],
        "min_samples_split": [params['minSamplesSplit']],
        "min_samples_leaf": [params['minSamplesLeaf']]
    }
    search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                                param_grid, n_iter=1, cv=tscv, scoring="r2", random_state=42)
    search.fit(rfXTrain, rfYTrain)
    rfModel = search.best_estimator_

    yPrediction = linearModel.predict(xTestFlat)
    rfPrediction = rfModel.predict(rfXTest)

    yPrediction = yPrediction.flatten()
    rfPrediction = rfPrediction.flatten()

    linearR2 = linearModel.score(xTestFlat, yTest)
    rfR2 = rfModel.score(rfXTest, rfYTest)

    totalR2 = linearR2 + rfR2

    if totalR2 > 0:
        linearWeight = linearR2 / totalR2
        rfWeight = rfR2 / totalR2
    else:
        linearWeight = 0.5
        rfWeight = 0.5
 
    minLength = min(len(yPrediction), len(rfPrediction))

    yPrediction = yPrediction[:minLength]
    rfPrediction = rfPrediction[:minLength]
    yTest = yTest[:minLength]

    combinedPrediction = linearWeight * yPrediction + rfWeight * rfPrediction

    lastSequence = y[-length:].copy()
    futurePrediction = []

    for i in range(futureAmount):
        pred = linearModel.predict(lastSequence.reshape(1, -1))
        futurePrediction.append(pred[0])
        
        lastSequence = np.append(lastSequence[1:], pred[0])

    rfLastSequence = rfData[-length:].copy()
    rfFuturePrediction = []

    for i in range(futureAmount):
        predRF = rfModel.predict(rfScaler.transform(rfLastSequence.reshape(1, -1)))
        rfFuturePrediction.append(predRF[0])
        
        newRow = rfLastSequence[-1].copy()
        newRow[3] = predRF[0] 
        rfLastSequence = np.append(rfLastSequence[1:], [newRow], axis=0)

    combinedFuturePrediction = [linearWeight * linear + rfWeight * rf for linear, rf in zip(futurePrediction, rfFuturePrediction)]



    # chart changes so plotting done on frontend

    plt.figure(figsize=(12, 6))
    plt.plot(stock.index, stock['Close'], label='Actual Price', color='blue')

    startIndex = length + splitIndex
    endIndex = startIndex + len(yPrediction)

    if endIndex <= len(stock.index):

        testDates = stock.index[startIndex:endIndex]
        testDates = testDates[:minLength]



        plt.scatter(testDates, yPrediction, color='red', label='Linear Predictions', alpha=0.6, s=20)
        plt.scatter(testDates, rfPrediction, color='green', label='RF Predictions', alpha=0.6, s=20)
        plt.scatter(testDates, combinedPrediction, color='purple', label='Combined Predictions', alpha=0.6, s=20)

    futureDates = pd.date_range(start=stock.index[-1] + pd.Timedelta(days=1), periods=futureAmount, freq='D')
    plt.plot(futureDates, futurePrediction, 'r--', label=f'Linear Future', linewidth=2)
    plt.plot(futureDates, rfFuturePrediction, 'g--', label=f'RF Future', linewidth=2)
    plt.plot(futureDates, combinedFuturePrediction, 'purple', linestyle='--', label=f'Combined Future', linewidth=2)


    plt.title(f'{ticker} Stock Price Prediction (Combined Ensemble)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    
    imgBuffer = io.BytesIO()
    plt.savefig(imgBuffer, format='png', dpi=300, bbox_inches='tight')
    imgBuffer.seek(0)

    chartBase64 = base64.b64encode(imgBuffer.getvalue()).decode()
    plt.close()  

    print(f"Linear R^2: {linearR2}")
    print(f"RF R^2: {rfR2}")
    print(f"Linear: {linearWeight:.3f}, RF: {rfWeight:.3f}")

    testStartIndex = length + splitIndex
    testDatesList = stock.index[testStartIndex:testStartIndex + minLength]
    
    historicalData = []
    for i, date in enumerate(testDatesList):
        historicalData.append({
            "date": date.strftime('%Y-%m-%d'),
            "actual": float(yTest[i]),
            "linearPrediction": float(yPrediction[i]),
            "rfPrediction": float(rfPrediction[i]),
            "combinedPrediction": float(combinedPrediction[i])
        })
    
    futureDatesList = pd.bdate_range(start=stock.index[-1] + pd.Timedelta(days=1), periods=futureAmount)
    
    futurePredictions = []
    for i, date in enumerate(futureDatesList):

        futurePredictions.append({

            "date": date.strftime('%Y-%m-%d'),
            "linearPrediction": float(futurePrediction[i]),
            "rfPrediction": float(rfFuturePrediction[i]),
            "combinedPrediction": float(combinedFuturePrediction[i])
        })

    
    return {
        "ticker": ticker,
        "historicalData": historicalData,
        "futurePredictions": futurePredictions,

        "metrics": {
            "linearR2": float(linearR2),
            "rfR2": float(rfR2),
            "linearWeight": float(linearWeight),
            "rfWeight": float(rfWeight)
        },

        "chart": chartBase64
    }




@app.route('/api/predict/<ticker>', methods=['POST'])
def predict_stock(ticker):
    ticker = ticker.upper()

    requestData = request.get_json() or {}
    params = getParameters(requestData)
    
    with requestLock:
        if ticker in activeRequests:
            print(f"Request for {ticker} already in progress, waiting...")
            event = activeRequests[ticker]
        else:
            activeRequests[ticker] = threading.Event()
            event = None

    if event:
        event.wait(timeout=120)  
        with requestLock:
            if ticker in activeRequests:
                del activeRequests[ticker]
        return jsonify({"error": "Request timeout - another request for this ticker is still processing"})
    
    try:
        print(f"Processing new request for {ticker} with params: {params}")
        result = predictPrice(ticker, params)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        with requestLock:
            if ticker in activeRequests:
                activeRequests[ticker].set()
                del activeRequests[ticker]

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

def signal_handler(sig, frame):
    with requestLock:
        for event in activeRequests.values():
            event.set()
        activeRequests.clear()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting server at http://localhost:5000")

    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server closed")