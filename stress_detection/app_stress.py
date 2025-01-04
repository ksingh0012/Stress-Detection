from flask import Flask, request, render_template, jsonify
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import nltk

# Initialize Flask app
app = Flask(__name__)

# Load the trained Logistic Regression model
try:
    with open('Logistic_Regression_stress_best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError("Logistic_Regression_stress_best_model.pkl not found. Ensure the model file is in the same directory.")

# Load the TF-IDF vectorizer
try:
    with open('tfidf_vectorizer_stress.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError:
    raise RuntimeError("tfidf_vectorizer_stress.pkl not found. Ensure the vectorizer file is in the same directory.")

# Initialize NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean input text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|[^a-zA-Z\s]', '', text)         # Remove mentions and special characters
    text = text.lower()                                  # Convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    """
    Renders the home page.
    """
    return render_template('index.html')  # Ensure `index.html` exists in the `templates` folder.

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')  # Get input text from the form

    # Validate the input
    if not text.strip():
        return jsonify({'error': 'No text provided for prediction.'}), 400

    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Check if cleaned text is empty
    if not cleaned_text:
        return jsonify({'error': 'Input text is empty after cleaning.'}), 400

    try:
        # Transform text using TF-IDF
        transformed_text = vectorizer.transform([cleaned_text])  

        # Predict sentiment using the Logistic Regression model
        prediction = model.predict(transformed_text)

        # Convert prediction to sentiment label
        sentiment = "Stress" if prediction[0] == 1 else "No Stress"

        return jsonify({'prediction': sentiment})
    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Use dynamic port assignment for better compatibility in hosted environments
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
