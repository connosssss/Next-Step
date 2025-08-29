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
    <div className="w-screen h-screen bg-emerald-400 text-black">
      
    
        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-red-200 p-2"
        >
          {loading ? 'Loading...' : 'Predict'}
        </button>

      {error && (
        <div className="bg-red-600">
          <div>Error:</div> {error}
        </div>
      )}

      
      <div className="bg-white p-4 rounded">
        <div className="text-xs overflow-auto">{JSON.stringify(stockData)}</div>
      </div>
      
    </div>
  );
}
