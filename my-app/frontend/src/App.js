import React, { useState } from 'react';
import logo from './imgs/grey_logo.png';
import './App.css';
import axios from 'axios';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [tokenList, setTokenList] = useState([]);
  const [numOutputs, setNumOutputs] = useState(5);

  const handleOutputChange = (event) => {
    setNumOutputs(parseInt(event.target.value));
  };

  const handleKeyPress = async (event) => {
    if (event.key === 'Enter') {
      try {
        const response = await axios.post('/api/predict', { input_string: searchQuery, num_outputs: numOutputs });
        console.log(response)
        setTokenList(response.data);
        console.log(tokenList)
      } catch (error) {
        console.error('Error sending search query:', error);
      }
    }
  };

  const handleChange = (event) => {
    setSearchQuery(event.target.value);
  };

  return (
    <div className="App">
      <img src={logo} className="App-logo" alt="logo" />
      <div className="instructions">
        <p>Run out of study material?</p>
        <p>Type a STEM homework problem below and we'll give you some more!</p>
        <div className='select-outputs'>
          <p>Select the desired number of outputs:</p>
          <select id="numberSelector" value={numOutputs} onChange={handleOutputChange}>
          {[...Array(10).keys()].map((num) => (
            <option key={num + 1} value={num + 1}>
              {num + 1}
            </option>
          ))}
          </select>
        </div>
      </div>
      <div className="searchBar">
        <input
          id="searchQueryInput"
          type="text"
          name="searchQueryInput"
          placeholder="Type a problem and hit Enter!"
          value={searchQuery}
          onChange={handleChange}
          onKeyPress={handleKeyPress}
        />
      </div>
      <div className="result-container">
          {tokenList.map((token, index) => (
            <div className='result-card' key={index}> 
              <p>{token}</p>
            </div>
          ))}
      </div>
    </div>
  );
}

export default App;
