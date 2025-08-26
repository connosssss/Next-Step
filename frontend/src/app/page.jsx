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
    
    try {
      const response = await fetch(`http://localhost:5000/api/predict/${ticker}`, {

        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})

      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      
      setStockData(data);
    } 
    
    catch (err) {
      console.error(err);
    }
    
    setLoading(false);
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
