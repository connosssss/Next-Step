import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


ticker = "AAPL"
stock = yf.download(ticker, period="1y")
y = stock['Close'].values


length = 60
futureAmount = 60 


def createSequences(data, length=60):
    X, y = [], []
    for i in range(length, len(data)):
        X.append(data[i-length:i])
        y.append(data[i])
    return np.array(X), np.array(y)



sequenceX, sequenceY = createSequences(y, length)
print(f"Sequences created: {len(sequenceX)} samples")


if len(sequenceX) == 0:
    print("Error: Not enough data to create sequences.")
    exit()

# splitting to training /testing sections
splitIndex = int(len(sequenceX) * 0.7)
xTrain, xTest = sequenceX[:splitIndex], sequenceX[splitIndex:]
yTrain, yTest = sequenceY[:splitIndex], sequenceY[splitIndex:]

print(f"Training samples: {len(xTrain)}, Test samples: {len(xTest)}")

xTrainFlat = xTrain.reshape(xTrain.shape[0], -1)
xTestFlat = xTest.reshape(xTest.shape[0], -1)

model = LinearRegression()
model.fit(xTrainFlat, yTrain)


yPrediction = model.predict(xTestFlat)
lastSequence = y[-length:].copy()
futurePrediction = []

for i in range(futureAmount):
    pred = model.predict(lastSequence.reshape(1, -1))
    futurePrediction.append(pred[0])
    

    lastSequence = np.append(lastSequence[1:], pred[0])



plt.figure(figsize=(12, 6))
plt.plot(stock.index, stock['Close'], label='Actual Price', color='blue')


startIndex = length + splitIndex
endIndex = startIndex + len(yPrediction)
if endIndex <= len(stock.index):
    test_dates = stock.index[startIndex:endIndex]
    plt.scatter(test_dates, yPrediction, color='red', label='Test Predictions', alpha=0.6, s=20)





futureDates = pd.date_range(start=stock.index[-1] + pd.Timedelta(days=1), periods=futureAmount, freq='D')
plt.plot(futureDates, futurePrediction, 'r--', label=f'Future Predictions', linewidth=2)


plt.title(f'{ticker} Stock Price Prediction (Sequence-based)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=300, bbox_inches='tight')
print(f"Model Score (RÂ²): {model.score(xTestFlat, yTest)}")
plt.show()