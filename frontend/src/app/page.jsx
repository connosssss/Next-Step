'use client';

import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
  Tooltip
} from 'chart.js';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip);

export default function Home() {

  const [ticker, setTicker] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [stockData, setStockData] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (loading) return;

    setLoading(true);
    setStockData(null);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5000/api/predict/${ticker}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        setError(data.error || `Error: ${response.status}`);
        return;
      }
      
      setStockData(data);
    } 
    catch (err) {
      console.error(err);
      setError('Network error or server unavailable');
    } 
    finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-screen min-h-screen h-full bg-emerald-300 text-emerald-700">
      <div className='p-12'>
      <h1 className='text-5xl font-bold '>Next Step</h1>
      <p className='text-xl py-6 w-1/3 font-semibold'> A stock market predictor that uses sckit to make linear regression and 
        random forest models, combining the results to get the most accurate prediction </p>
      <div className='w-full flex justify-start mb-4 gap-5 items-center'>

        <textarea className='bg-emerald-100/70 h-12 rounded-md items-center text-xl resize-none text-center '
        placeholder='AAPL'
        onChange={e => setTicker(e.target.value)}
        cols={9}/>

        
        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-emerald-100/70 px-4 py-2 text-2xl font-bold rounded-xl"
        >
          {loading ? 'Loading...' : 'Predict'}
        </button>
        {loading && (
          <div className=' w-full  flex items-center '>
            <p> The models train after in real time, may take a while to load</p>
            </div>
        )}

      {error && (
          <div className="bg-red-200/60 text-lg text-red-400/80 font-bold p-2 rounded-lg">
            <div>Error:</div> {error}
          </div>
        )}

      </div>
        <div className="p-4 rounded">
          {stockData && stockData.historicalData && (
            <div className='flex flex-col text-center text-4xl w-full font-bold'>
              Training on Past Data
            <Line
              className='rounded-xl bg-emerald-200 p-10 mt-5 mb-12'
              data={{
                labels: stockData.historicalData.map(d => d.date),
                datasets: [
                  {
                    label: 'Actual',
                    data: stockData.historicalData.map(d => d.actual),
                    borderColor: 'blue',
                    fill: false
                  },
                  {
                    label: 'Combined Prediction',
                    data: stockData.historicalData.map(d => d.combinedPrediction),
                    borderColor: 'purple',
                    fill: false
                  },
                  {
                    label: 'Linear Prediction',
                    data: stockData.historicalData.map(d => d.linearPrediction),
                    borderColor: 'red',
                    fill: false
                  },
                  {
                    label: 'Random Forest Prediction',
                    data: stockData.historicalData.map(d => d.rfPrediction),
                    borderColor: 'green',
                    fill: false
                  }
                ]
              }}

              options={{
                responsive: true,
                plugins: {
                  legend: { display: true }
                },

                scales: {
                  x: { display: true },
                  y: { display: true }
                }
              }}
            />
            Future Predictions
            <Line
            className='rounded-xl bg-emerald-200 p-10 mt-5'
              data={{
                labels: stockData.futurePredictions.map(d => d.date),
                datasets: [
                  {
                    label: 'Combined Prediction',
                    data: stockData.futurePredictions.map(d => d.combinedPrediction),
                    borderColor: 'purple',
                    fill: false
                  },
                  {
                    label: 'Linear Prediction',
                    data: stockData.futurePredictions.map(d => d.linearPrediction),
                    borderColor: 'red',
                    fill: false
                  },
                  {
                    label: 'Random Forest Prediction',
                    data: stockData.futurePredictions.map(d => d.rfPrediction),
                    borderColor: 'green',
                    fill: false
                  }
                ]
              }}

              options={{
                responsive: true,
                plugins: {
                  legend: { display: true }
                },

                scales: {
                  x: { display: true },
                  y: { display: true }
                }
              }}
            />
            </div>
          )}
          
        </div>
      
      </div>
    </div>
  );
}
