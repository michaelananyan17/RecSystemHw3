// Global variables
let model;
let trainingData = {};
let isModelTrained = false;

// Initialize when window loads
window.onload = async function() {
    try {
        // Load and parse data
        trainingData = await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('result').innerHTML = 
            '<p class="error">Error loading data or training model. Check console for details.</p>';
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Get unique users from ratings and sort numerically
    const uniqueUsers = [...new Set(trainingData.ratings.map(r => r.userId))].sort((a, b) => a - b);
    
    // CHANGED: Users are now sorted numerically from least to greatest
    uniqueUsers.forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        userSelect.appendChild(option);
    });
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // CHANGED: Movies are now sorted alphabetically by title from data.js
    trainingData.movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = `${movie.id}: ${movie.title}`;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 10) {
    // Input layers
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // EMBEDDING LAYERS
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers + 1, // +1 because user IDs start from 1
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies + 1, // +1 because movie IDs start from 1
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // LATENT VECTORS
    // Flatten the embeddings to get 1D latent vectors
    const userLatent = tf.layers.flatten().apply(userEmbedding);
    const movieLatent = tf.layers.flatten().apply(movieEmbedding);
    
    // PREDICTION
    // Compute dot product of user and movie latent vectors
    const dotProduct = tf.layers.dot({axes: 1}).apply([userLatent, movieLatent]);
    
    // Add bias terms (optional but improves performance)
    const userBias = tf.layers.embedding({
        inputDim: numUsers + 1,
        outputDim: 1,
        name: 'userBias'
    }).apply(userInput);
    
    const movieBias = tf.layers.embedding({
        inputDim: numMovies + 1,
        outputDim: 1,
        name: 'movieBias'
    }).apply(movieInput);
    
    const userBiasFlat = tf.layers.flatten().apply(userBias);
    const movieBiasFlat = tf.layers.flatten().apply(movieBias);
    
    // Combine dot product with biases
    const prediction = tf.layers.add().apply([
        dotProduct, 
        userBiasFlat, 
        movieBiasFlat
    ]);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: prediction
    });
    
    console.log('Matrix Factorization model created successfully');
    return model;
}

async function trainModel() {
    try {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<p class="loading">Creating model architecture...</p>';
        
        // Create model
        model = createModel(trainingData.numUsers, trainingData.numMovies, 10);
        
        // Compile model
        resultDiv.innerHTML = '<p class="loading">Compiling model...</p>';
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        // Prepare training data
        resultDiv.innerHTML = '<p class="loading">Preparing training data...</p>';
        
        const userIds = trainingData.ratings.map(r => r.userId);
        const movieIds = trainingData.ratings.map(r => r.movieId);
        const ratings = trainingData.ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratings, [ratings.length, 1]);
        
        // Train model
        resultDiv.innerHTML = '<p class="loading">Training model (this may take a minute)...</p>';
        
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 8,
            batchSize: 128,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    resultDiv.innerHTML = `<p class="loading">Training epoch ${epoch + 1}/8 - Loss: ${logs.loss.toFixed(4)}</p>`;
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        // Enable predict button and update UI
        document.getElementById('predict-btn').disabled = false;
        isModelTrained = true;
        resultDiv.innerHTML = '<p class="success">Model trained successfully! Select a user and movie to predict ratings.</p>';
        
        console.log('Model training completed');
        
    } catch (error) {
        console.error('Training error:', error);
        document.getElementById('result').innerHTML = 
            '<p class="error">Error training model. Check console for details.</p>';
    }
}

async function predictRating() {
    if (!isModelTrained) {
        alert('Model is still training. Please wait...');
        return;
    }
    
    const userId = parseInt(document.getElementById('user-select').value);
    const movieId = parseInt(document.getElementById('movie-select').value);
    
    if (!userId || !movieId) {
        alert('Please select both a user and a movie.');
        return;
    }
    
    try {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<p class="loading">Making prediction...</p>';
        
        // Create input tensors
        const userTensor = tf.tensor2d([userId], [1, 1]);
        const movieTensor = tf.tensor2d([movieId], [1, 1]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const predictedRating = await prediction.data();
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Get movie title
        const movie = trainingData.movies.find(m => m.id === movieId);
        const movieTitle = movie ? movie.title : `Movie ${movieId}`;
        
        // Display result
        const roundedRating = Math.min(5, Math.max(1, Math.round(predictedRating[0] * 10) / 10));
        resultDiv.innerHTML = `
            <p><strong>Prediction Result:</strong></p>
            <p>User ${userId} would rate "${movieTitle}"</p>
            <p class="prediction">${roundedRating.toFixed(1)} / 5.0 stars</p>
        `;
        
    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('result').innerHTML = 
            '<p class="error">Error making prediction. Check console for details.</p>';
    }
}