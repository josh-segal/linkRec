import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import './App.css';

function App() {
  const [modelType, setModelType] = useState('linucb');
  const [context, setContext] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [weights, setWeights] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [articleDetails, setArticleDetails] = useState({});
  const [hoveredArticle, setHoveredArticle] = useState(null);
  const [recommendationScores, setRecommendationScores] = useState({});

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);
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
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setRecommendations(data.recommendations);
      setContext(data.context);
      
      // Store the scores for each recommendation
      const scores = {};
      const details = {};
      data.recommendations.forEach((armIndex, i) => {
        scores[armIndex] = data.scores[i];
        details[armIndex] = data.article_details[i];
      });
      setRecommendationScores(scores);
      setArticleDetails(details);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError('Failed to fetch recommendations');
    } finally {
      setLoading(false);
    }
  };

  const updateModel = async (armIndex, reward) => {
    try {
      setLoading(true);
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
      
      await fetchWeights();
      await fetchRecommendations();
    } catch (error) {
      console.error('Error updating model:', error);
      setError('Failed to update model');
    } finally {
      setLoading(false);
    }
  };

  const fetchWeights = async () => {
    try {
      const response = await fetch(`http://localhost:8000/model_weights/${modelType}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setWeights(data.weights);
    } catch (error) {
      console.error('Error fetching weights:', error);
      setError('Failed to fetch weights');
    }
  };

  const resetModel = async () => {
    try {
      setLoading(true);
      await fetch(`http://localhost:8000/reset/${modelType}`, {
        method: 'POST',
      });
      await fetchWeights();
      await fetchRecommendations();
    } catch (error) {
      console.error('Error resetting model:', error);
      setError('Failed to reset model');
    } finally {
      setLoading(false);
    }
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
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      <div className="controls">
        <select 
          value={modelType} 
          onChange={(e) => setModelType(e.target.value)}
          disabled={loading}
        >
          <option value="linucb">LinUCB</option>
          <option value="thompson">Thompson Sampling</option>
          <option value="doubly_robust">Doubly Robust Neural</option>
          <option value="slate_ranking">Slate Ranking Neural</option>
        </select>
        <button onClick={resetModel} disabled={loading}>Reset Model</button>
      </div>

      <div className="recommendations">
        <h2>Recommendations</h2>
        {loading ? (
          <div>Loading...</div>
        ) : (
          recommendations.map((armIndex, i) => (
            <div key={armIndex} className="recommendation">
              <div 
                className="article-button"
                onClick={() => updateModel(armIndex, 1)}
                onMouseEnter={() => setHoveredArticle(armIndex)}
                onMouseLeave={() => setHoveredArticle(null)}
                style={{ cursor: 'pointer' }}
              >
                <span className="article-title">{articleDetails[armIndex]?.title || `Article ${armIndex}`}</span>
                <span className="article-score">(Score: {recommendationScores[armIndex]?.toFixed(3) || 'N/A'})</span>
              </div>
              {hoveredArticle === armIndex && (
                <div className="article-abstract">
                  {articleDetails[armIndex]?.abstract || 'No abstract available'}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {modelType !== 'doubly_robust' && modelType !== 'slate_ranking' && (
        <div className="visualization">
          <h2>Model Weights</h2>
          <BarChart width={500} height={300} data={weightData}>
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
      )}
    </div>
  );
}

export default App;