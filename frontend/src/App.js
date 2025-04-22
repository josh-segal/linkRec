import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import './App.css';

function App() {
  const [modelType, setModelType] = useState('linucb');
  const [context, setContext] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [weights, setWeights] = useState({});
  const [performance, setPerformance] = useState({});

  // Fetch recommendations
  const fetchRecommendations = async () => {
    const response = await fetch('http://localhost:8000/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_type: modelType,
        k: 3
      }),
    });
    const data = await response.json();
    setRecommendations(data.recommendations);
    setContext(data.context);
  };

  // Update model with feedback
  const updateModel = async (armIndex, reward) => {
    await fetch('http://localhost:8000/update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_type: modelType,
        chosen_arm: armIndex,
        reward: reward
      }),
    });
    fetchWeights();
    fetchRecommendations();
  };

  // Fetch model weights
  const fetchWeights = async () => {
    const response = await fetch(`http://localhost:8000/model_weights/${modelType}`);
    const data = await response.json();
    setWeights(data.weights);
    setPerformance(data.performance);
  };

  // Reset model
  const resetModel = async () => {
    await fetch(`http://localhost:8000/reset/${modelType}`, {
      method: 'POST',
    });
    fetchWeights();
    fetchRecommendations();
  };

  useEffect(() => {
    fetchRecommendations();
    fetchWeights();
  }, [modelType]);

  // Prepare data for visualization
  const weightData = Object.entries(weights).map(([arm, values]) => ({
    arm: `Arm ${arm}`,
    ...values.reduce((acc, val, idx) => ({
      ...acc,
      [`weight_${idx}`]: val
    }), {})
  }));

  return (
    <div className="App">
      <h1>Bandit Model Interface</h1>
      
      <div className="controls">
        <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
          <option value="linucb">LinUCB</option>
          <option value="thompson">Thompson Sampling</option>
        </select>
        <button onClick={resetModel}>Reset Model</button>
      </div>

      <div className="recommendations">
        <h2>Recommendations</h2>
        {recommendations.map((armIndex, i) => (
          <div key={armIndex} className="recommendation">
            <span>Arm {armIndex}</span>
            <button onClick={() => updateModel(armIndex, 1)}>👍</button>
            <button onClick={() => updateModel(armIndex, 0)}>👎</button>
          </div>
        ))}
      </div>

      <div className="performance">
        <h2>Performance</h2>
        <p>Average Reward: {performance.average_reward?.toFixed(3)}</p>
        <p>Total Trials: {performance.n_trials}</p>
      </div>

      <div className="visualization">
        <h2>Model Weights</h2>
        <BarChart width={600} height={300} data={weightData}>
          <XAxis dataKey="arm" />
          <YAxis />
          <Tooltip />
          <Legend />
          {Object.keys(weightData[0] || {})
            .filter(key => key.startsWith('weight_'))
            .map((key, i) => (
              <Bar key={key} dataKey={key} fill={`hsl(${i * 30}, 70%, 50%)`} />
            ))}
        </BarChart>
      </div>
    </div>
  );
}

export default App;