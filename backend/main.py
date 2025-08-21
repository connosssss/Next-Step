import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import pandas as pd



ticker = "AAPL"
stock = yf.download(ticker, period="2y")
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

rfScaler = StandardScaler()
rfXTrain = rfScaler.fit_transform(rfXTrainFlat)
rfXTest = rfScaler.transform(rfXTestFlat)

linearModel = LinearRegression()
linearModel.fit(xTrainFlat, yTrain)

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [10, 20, None],
    "max_features": [0.4, 0.6, "sqrt"],
    "min_samples_split": [2, 8],
    "min_samples_leaf": [1, 3, 5]
}
search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                            param_grid, n_iter=10, cv=tscv, scoring="r2", random_state=42)
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

plt.figure(figsize=(12, 6))
plt.plot(stock.index, stock['Close'], label='Actual Price', color='blue')


startIndex = length + splitIndex
endIndex = startIndex + len(yPrediction)
if endIndex <= len(stock.index):
    test_dates = stock.index[startIndex:endIndex]

    test_dates = test_dates[:minLength]

    plt.scatter(test_dates, yPrediction, color='red', label='Test Predictions', alpha=0.6, s=20)
    plt.scatter(test_dates, rfPrediction, color='green', label='Test Predictions', alpha=0.6, s=20)
    plt.scatter(test_dates, combinedPrediction, color='blue', label='Combined Predictions', alpha=0.6, s=20)

futureDates = pd.date_range(start=stock.index[-1] + pd.Timedelta(days=1), periods=futureAmount, freq='D')
plt.plot(futureDates, futurePrediction, 'r--', label=f'Future Predictions', linewidth=2)
plt.plot(futureDates, rfFuturePrediction, 'g--', label=f'Random Forest Future', linewidth=2)
plt.plot(futureDates, combinedFuturePrediction, 'b--', label=f'Combined Future', linewidth=2)


plt.title(f'{ticker} Stock Price Prediction (Combined Ensemble)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=300, bbox_inches='tight')



print(f"Linear R^2: {linearR2}")
print(f"RF R^2: {rfR2}")
print(f"Linear: {linearWeight:.3f}, RF: {rfWeight:.3f}")
plt.show()