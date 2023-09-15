from flask import Flask, request, jsonify
import PredictGenre 
from keras.models import load_model
from flask_cors import CORS




app = Flask(__name__)
CORS(app)
model = load_model('./src/final_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the YouTube URL from the request
        url = request.json['url']

        # Download the audio from the YouTube URL (you can use your previous download function)
        # Process the audio (e.g., extract features, preprocess)
        # Make predictions using your classifier model
        # Return the predictions as JSON
        song = PredictGenre.download_audio_from_youtube(url)
        predictions = PredictGenre.classify(*PredictGenre.load_audio_with_librosa(song))
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
