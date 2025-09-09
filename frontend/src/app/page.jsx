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
  const [showAdvanced, setShowAdvanced] = useState(false);


  const [parameters, setParameters] = useState({
    sequenceLength: 60,
    futureDays: 60,
    trainSplit: 0.7,
    nEstimators: 500,
    maxDepth: 20,
    maxFeatures: 0.6,
    minSamplesSplit: 2,
    minSamplesLeaf: 1
  });

  const handleParameterChange = (param, value) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }));
  };

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
        body: JSON.stringify(parameters)
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

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="bg-emerald-100/70 px-4 py-2 text-xl font-bold rounded-xl"
        >
          {showAdvanced ? 'Hide' : 'Show'} Settings
        </button>

        {loading && (
          <div className=' w-full  flex items-center gap-4'>
            <p> The models train after in real time, may take a while to load</p> 
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-700"/>
          </div>
        )}


        {error && (<div className="bg-red-200/60 text-lg text-red-400/80 font-bold p-2 rounded-lg">
            <div>Error:</div> {error}
          </div>
        )}

      </div>

      {showAdvanced && (
        <div className="bg-emerald-200/50 p-6 rounded-xl mb-6">
          <div className="grid grid-cols-2 gap-6">

            <div className="space-y-4">
              <h4 className="text-lg font-semibold">General Settings</h4>

              <div className="flex items-center gap-3">
                <label className="w-36">Sequence Length:</label>

                <input
                  type="number"
                  value={parameters.sequenceLength}
                  onChange={e => handleParameterChange('sequenceLength', parseInt(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="10"
                  max="200"
                /> 


                  <span className="text-sm opacity-70">days of historical data per prediction</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="w-36 ">Future Days:</label>

                <input
                  type="number"
                  value={parameters.futureDays}
                  onChange={e => handleParameterChange('futureDays', parseInt(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="1"
                  max="365"
                />


                  <span className="text-sm opacity-70">days to predict into future</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="w-36 ">Train Split:</label>

                <input
                  type="number"
                  step="0.1"
                  value={parameters.trainSplit}
                  onChange={e => handleParameterChange('trainSplit', parseFloat(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="0.1"
                  
                  
                  max="0.9"
                />  


                <span className="text-sm opacity-70">fraction of data for training</span>
              </div>
            </div>


            <div className="space-y-4">
              <h4 className="text-lg font-semibold">Random Forest Specific</h4>
              
              <div className="flex items-center gap-3">
                <label className="w-36 ">N Estimators:</label>

                <input
                  type="number"
                  value={parameters.nEstimators}
                  onChange={e => handleParameterChange('nEstimators', parseInt(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="50"
                  max="2000"
                  
                  
                  step="50"
                />  


                <span className="text-sm opacity-70">number of trees</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="w-36 ">Max Depth:</label>

                <input
                  type="number"
                  value={parameters.maxDepth}
                  onChange={e => handleParameterChange('maxDepth', parseInt(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="5"
                  max="50"
                />


                  <span className="text-sm opacity-70">maximum tree depth</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="w-36 ">Max Features:</label>

                <input
                  type="number"
                  step="0.1"
                  value={parameters.maxFeatures}
                  onChange={e => handleParameterChange('maxFeatures', parseFloat(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="0.1"
                  
                  
                  max="1.0"
                />  

                <span className="text-sm opacity-70">fraction of features to consider</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="w-36 ">Min Samples Split:</label>

                <input
                  type="number"
                  value={parameters.minSamplesSplit}
                  onChange={e => handleParameterChange('minSamplesSplit', parseInt(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="2"
                  max="20"
                />


                  <span className="text-sm opacity-70">min samples to split node</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="w-36 ">Min Samples Leaf:</label>

                <input
                  type="number"
                  value={parameters.minSamplesLeaf}
                  onChange={e => handleParameterChange('minSamplesLeaf', parseInt(e.target.value))}
                  className="bg-emerald-100/70 px-2 py-1 rounded w-20 text-center"
                  min="1"
                  max="10"
                />


                  <span className="text-sm opacity-70">min samples in leaf node</span>
              </div>
            </div>
          </div>
        </div>
      )}

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
            
            {stockData.metrics && (
              <div className="text-lg mt-6 bg-emerald-100/50 p-4 rounded-lg">
                <div className="grid grid-cols-2 gap-4 text-left">
                  <div>Linear R²: {stockData.metrics.linearR2.toFixed(4)}</div>
                  <div>Random Forest R²: {stockData.metrics.rfR2.toFixed(4)}</div>
                  <div>Linear Weight: {stockData.metrics.linearWeight.toFixed(3)* 100 + " "} %</div>
                  <div>RF Weight: {stockData.metrics.rfWeight.toFixed(3) * 100 + " "} %</div>
                </div>
              </div>
            )}
            </div>
          )}
          
        </div>
      
      </div>
    </div>
  );
}
