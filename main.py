import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import pandas as pd



ticker = "AAPL"
stock = yf.download(ticker, period="1y")
y = stock['Close'].values


stock['Price_Range'] = stock['High'] - stock['Low']
stock['Volume_Price'] = stock['Volume'] * stock['Close']
stock['Adj_Close_MA3'] = stock['Close'].rolling(window=3).mean()

rfData = stock[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Range', 'Volume_Price', 'Adj_Close_MA3']].fillna(method='bfill').values


length = 60
futureAmount = 60 


def createSequences(data, length=60):
    X, y = [], []
    for i in range(length, len(data)):
        X.append(data[i-length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def createRFSequences(data, target, length=60):
    X, y = [], []
    for i in range(length, len(data)):
        X.append(data[i-length:i])
        y.append(target[i])
    return np.array(X), np.array(y)


sequenceX, sequenceY = createSequences(y, length)
rfSequenceX, rfSequenceY = createRFSequences(rfData, y, length)
print(f"Sequences created: {len(sequenceX)} samples")


if len(sequenceX) == 0:
    print("Error: Not enough data to create sequences.")
    exit()

# splitting to training /testing sections
splitIndex = int(len(sequenceX) * 0.7)
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



linearModel = LinearRegression()
linearModel.fit(xTrainFlat, yTrain)

rfModel = RandomForestRegressor(n_estimators=1000, random_state=42,
                                max_features=0.6, n_jobs=-1,
                                 min_samples_leaf=3, min_samples_split=8,
                                 max_depth=20 )
rfModel.fit(rfXTrainFlat, rfYTrain)


yPrediction = linearModel.predict(xTestFlat)
rfPrediction = rfModel.predict(rfXTestFlat)
lastSequence = y[-length:].copy()
futurePrediction = []

for i in range(futureAmount):
    pred = linearModel.predict(lastSequence.reshape(1, -1))
    futurePrediction.append(pred[0])
    

    lastSequence = np.append(lastSequence[1:], pred[0])


rfLastSequence = rfData[-length:].copy()
rfFuturePrediction = []

for i in range(futureAmount):
    predRF = rfModel.predict(rfLastSequence.reshape(1, -1))
    rfFuturePrediction.append(predRF[0])
    
    newRow = rfLastSequence[-1].copy()
    newRow[3] = predRF[0] 
    rfLastSequence = np.append(rfLastSequence[1:], [newRow], axis=0)

plt.figure(figsize=(12, 6))
plt.plot(stock.index, stock['Close'], label='Actual Price', color='blue')


startIndex = length + splitIndex
endIndex = startIndex + len(yPrediction)
if endIndex <= len(stock.index):
    test_dates = stock.index[startIndex:endIndex]
    plt.scatter(test_dates, yPrediction, color='red', label='Test Predictions', alpha=0.6, s=20)
    plt.scatter(test_dates, rfPrediction, color='green', label='Test Predictions', alpha=0.6, s=20)





futureDates = pd.date_range(start=stock.index[-1] + pd.Timedelta(days=1), periods=futureAmount, freq='D')
plt.plot(futureDates, futurePrediction, 'r--', label=f'Future Predictions', linewidth=2)
plt.plot(futureDates, rfFuturePrediction, 'g--', label=f'Random Forest Future', linewidth=2)

plt.title(f'{ticker} Stock Price Prediction (Sequence-based)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=300, bbox_inches='tight')
print(f"Linear R^2: {linearModel.score(xTestFlat, yTest)}")
print(f"RF R^2: {rfModel.score(rfXTestFlat, rfYTest)}")
plt.show()