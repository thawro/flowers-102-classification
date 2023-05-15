import React, { useState, useEffect } from "react";
import axios from 'axios';

import './App.css';

function ImageUploader() {
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setSelectedImage(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    // Make a POST request to the FastAPI backend
    axios.post('http://0.0.0.0:5000/predict', formData)
      .then(response => {
        // Handle successful response
        console.log(response.data);
      })
      .catch(error => {
        // Handle error
        console.error(error);
      });
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {selectedImage && (
        <img src={selectedImage} alt="Selected" style={{ width: '200px', height: '200px' }} />
      )}
    </div>
  );
}






function ExampleComponent() {
  const [data, setData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch('http://0.0.0.0:5000/');
        const data = await response.json();
        setData(data);
        console.log(data)
      } catch (error) {
        console.error(error);
      }
    }
    fetchData();
  }, []);

  return (
    <div>
      {data ? <p>{data.example}</p> : <p>Loading...</p>}
    </div>
  );
}





function App() {
  return ImageUploader()
}

export default App;
