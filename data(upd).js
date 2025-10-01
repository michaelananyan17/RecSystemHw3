// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// MovieLens dataset URLs
const MOVIES_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/multivariate-linear-regression/data/u.item';
const RATINGS_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/multivariate-linear-regression/data/u.data';

async function loadData() {
    try {
        console.log('Loading movie data...');
        const moviesResponse = await fetch(MOVIES_URL);
        const moviesText = await moviesResponse.text();
        movies = parseItemData(moviesText);
        
        console.log('Loading rating data...');
        const ratingsResponse = await fetch(RATINGS_URL);
        const ratingsText = await ratingsResponse.text();
        ratings = parseRatingData(ratingsText);
        
        console.log(`Data loaded: ${movies.length} movies, ${ratings.length} ratings`);
        
        return {
            movies: movies,
            ratings: ratings,
            numUsers: numUsers,
            numMovies: numMovies
        };
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

function parseItemData(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const movieMap = new Map();
    
    lines.forEach(line => {
        const parts = line.split('|');
        if (parts.length >= 2) {
            const movieId = parseInt(parts[0]);
            const title = parts[1];
            if (!isNaN(movieId) && title) {
                movieMap.set(movieId, {
                    id: movieId,
                    title: title
                });
            }
        }
    });
    
    numMovies = movieMap.size;
    console.log(`Parsed ${numMovies} movies`);
    
    // CHANGED: Convert Map to array and sort by title alphabetically
    return Array.from(movieMap.values()).sort((a, b) => {
        // Remove any leading/trailing whitespace and compare case-insensitively
        const titleA = a.title.trim().toLowerCase();
        const titleB = b.title.trim().toLowerCase();
        return titleA.localeCompare(titleB);
    });
}

function parseRatingData(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const ratingData = [];
    const userSet = new Set();
    
    lines.forEach(line) => {
        const parts = line.split('\t');
        if (parts.length >= 3) {
            const userId = parseInt(parts[0]);
            const movieId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            
            if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating)) {
                ratingData.push({
                    userId: userId,
                    movieId: movieId,
                    rating: rating
                });
                userSet.add(userId);
            }
        }
    });
    
    numUsers = userSet.size;
    console.log(`Parsed ${ratingData.length} ratings from ${numUsers} users`);
    
    return ratingData;
}