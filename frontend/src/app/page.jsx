'use client';

import React, { useState } from 'react';


// Temporarily switching to javascript instead of typescript for testing purposes

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
    <div className="w-screen h-screen bg-emerald-300 text-emerald-700">
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
          <div className="bg-red-600">
            <div>Error:</div> {error}
          </div>
        )}

      </div>
        <div className="bg-white p-4 rounded">
          <div className="text-xs overflow-auto">{JSON.stringify(stockData)}</div>
        </div>
      
      </div>
    </div>
  );
}
