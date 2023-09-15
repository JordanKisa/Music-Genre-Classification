import React, { useState, useEffect } from 'react';
import './App.css';
import { RingLoader } from 'react-spinners';

import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import randomColor from 'randomcolor';

ChartJS.register(ArcElement, Tooltip, Legend);

const chartContainerStyle = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  maxHeight: '50vh', // Set the maximum height to 90% of the viewport height
};

const PieChart = ({ data }) => {
  const [chartData, setChartData] = useState(null);
  const randomColors = randomColor({ count: data.length });

  useEffect(() => {
    if (data) {
      // Extract labels and frequencies from the data
      const labels = data.map(item => item[0]);
      const frequencies = data.map(item => item[1]);

      setChartData({
        labels: labels,
        datasets: [
          {
            data: frequencies,
            backgroundColor: randomColors
          },
        ],
      });
    }
  }, [data]);

  const options = {
    plugins: {
      legend: {
        display: true,
        position: 'right',
      },
      tooltip: {
        enabled: false, // Disable tooltip
      },
      datalabels: { // Display labels on slices
        formatter: (value, context) => {
          const label = context.chart.data.labels[context.dataIndex];
          return label + ': ' + value;
        },
        color: 'black', // Label text color
        align: 'center', // Label text alignment
        anchor: 'end', // Label text anchor
        offset: 0, // Offset from the center of the slice
        font: {
          size: 12, // Label text font size
          weight: 'bold', // Label text font weight
        },
      },
    },
  };

  return (
    <div style={chartContainerStyle}>
      {chartData && <Doughnut data={chartData}/>}
    </div>
  );
};

// Rest of the code remains the same



function App() {
  const [url, setUrl] = useState('');
  const [response, setResponse] = useState('');

  const handleUrlChange = (event) => {
    setUrl(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      setResponse("Loading")
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      console.log(data);
      setResponse(data);
    } catch (error) {
      console.error(error);
      setResponse('An error occurred.');
    }
  };

  return (
    <div className="App">
      <div className='header'></div>
      <h1>YouTube URL to API</h1>
      <form onSubmit={handleSubmit}>
      YouTube URL:
        <label>
          <input type="text" value={url} onChange={handleUrlChange} />
        </label>
        <button type="submit">Submit</button>
      </form>
      <div className="response">
        <h2>API Response:</h2>
        <div style={chartContainerStyle}>
          {(response === "Loading" &&  <RingLoader color={'#123abc'} size={150} />) || (response !== null && response !== 'An error occurred.' && <PieChart data={response} />)}
        </div>
        
      </div>
    </div>
  );
}

export default App;

