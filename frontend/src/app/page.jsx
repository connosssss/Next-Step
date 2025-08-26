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
      
      
      setStockData(data);
    } 
    
    catch (err) {
      console.error(err);
      setError('Network error or server unavailable');
    } finally {
      setLoading(false);
    }
    
                    
  };

  return (
    <div className="w-screen h-screen bg-emerald-400 text-black">
      {JSON.stringify(stockData)}

      <button
        onClick={handlePredict}
        className=" bg-red-200"
      > Test</button>
    </div>
  );
}
