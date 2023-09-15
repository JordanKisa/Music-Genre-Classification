import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import warnings
from pytube import YouTube
from pydub import AudioSegment
from io import BytesIO

# Disable all warnings (not recommended in most cases)
warnings.filterwarnings("ignore")

model = load_model('./models/final_model.keras')
mp3 = "data/Amapiano/Extra/Kweyama Brothers, Mpura, Abidoza, Thabiso Lavish - Impilo yaseSandton.mp3"


y, sr = librosa.load(mp3)
parent_folder = os.path.basename(os.path.dirname(os.path.dirname(mp3)))


def extract_audio_features(y,sr):
    """
    Extract audio features from an audio file.
    
    Args:
        y sr (int): Audio_data and Sample Rate

    Returns:
        audio_features (list): List of audio features.
    """
    print("Extracting audio features")
    desired_duration = 60  # seconds
    desired_samples = int(desired_duration * sr)

# Crop the audio to the desired duration
    y = y[:desired_samples]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_variance = np.var(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)
    chroma_var = np.var(chroma)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength_mean = np.mean(onset_strength)
    onset_strength_var = np.var(onset_strength)

    harmonic_percussive = librosa.effects.hpss(y)

    harmonic = harmonic_percussive[0]
    harmonic_mean = np.mean(harmonic)
    harmonic_var = np.var(harmonic)

    percussive = harmonic_percussive[1]
    percussive_mean= np.mean(percussive)
    percussive_var = np.var(percussive)

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz)
    tonnetz_var = np.var(tonnetz)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Combine all the features
    audio_features = [tempo, chroma_mean, chroma_var, spectral_centroid_mean,  spectral_centroid_var,
            zero_crossing_rate_mean, zero_crossing_rate_var,  onset_strength_mean, onset_strength_var,
            harmonic_mean, harmonic_var, percussive_mean, percussive_var, tonnetz_mean, tonnetz_var, rms_mean, rms_var]
    audio_features.extend(list(mfcc_mean))
    audio_features.extend(list(mfcc_variance))

    audio_features = np.expand_dims(audio_features, axis=0)

    scaler = joblib.load('./src/minmax_scaler.pkl')
    audio_features = scaler.transform(audio_features)
    
    return audio_features

def extract_mel_spectrograms(y, sr):
    """
    Extract mel spectrogram images from an audio file.

    Args:
        y sr (int): Audio_data and Sample Rate

    Returns:
        mel_spectrograms (list): List of mel spectrogram images.
    """
    print("Extracting mel spectrograms")
    segment_duration = 3
    n_frames_per_segment = int(segment_duration * sr) 
    num_segments = int(np.floor(len(y) / n_frames_per_segment))

    data = []

    for i in range(num_segments):
        
        start_frame = i * n_frames_per_segment
        end_frame = min((i + 1) * n_frames_per_segment, len(y))
    
   
        segment = y[start_frame:end_frame]
    
        mel = librosa.feature.melspectrogram(y=segment, sr=sr)
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        plt.imsave("temp.png", mel_db, cmap='Greys')

        image = cv2.imread("temp.png", 0)
        image = cv2.resize(image, (131,128))
        image_arr = img_to_array(image)
        image_arr = image_arr / 255.0
        data.append(image_arr)
    
    return np.array(data)

def predict_genre(model, audio_features, mel_spectrograms):
    """
    Predict the genre of audio segments.

    Args:
        model (object): Pretrained Keras model.
        audio_features (list): List of audio features.
        mel_spectrograms (numpy.ndarray): Array of mel spectrogram images.

    Returns:
        predicted_labels (numpy.ndarray): Predicted genre labels.
    """


    # Predict genre for each segment
    predictions = model.predict([mel_spectrograms, np.tile(audio_features, (mel_spectrograms.shape[0], 1))])
    predicted_labels = np.argmax(predictions, axis=1)

    label_mapping = {'Funky-Soul': 0, 'Favela-Funk': 1, 'Samba': 2,
                     'Amapiano': 3, 'Jungle': 4, 'Reggae': 5, 
                     'Soca': 6, 'Bossa-Nova': 7, 'RnB': 8,
                     'Neo-Soul': 9, 'Reggaeton': 10}
    index_to_label = {index: label for label, index in label_mapping.items()}
    pred_labels_class = [index_to_label[label] for label in predicted_labels]

    unique_values = set(pred_labels_class)

    count_dict = {value: pred_labels_class.count(value) for value in unique_values}
    sorted_values = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_values

def print_genre_predictions(predicted_counts):
    """
    Print genre predictions with likelihood percentages.

    Args:
        predicted_labels (numpy.ndarray): Predicted genre labels.
    """
    total_sum = 0
    for key, value in predicted_counts:
        total_sum += value

    for value, count in predicted_counts:
        percentage = (count / total_sum) * 100
        print(f"Genre: {value}, Likelihood: {percentage:.2f}%")

        
def classify(y, sr):

    audio_features = extract_audio_features(y, sr)
    mel_spectrograms = extract_mel_spectrograms(y, sr)
    
    predicted_labels = predict_genre(model, audio_features, mel_spectrograms)
    print_genre_predictions(predicted_labels)
    return predicted_labels
    
def main():
    mp3 = "data/Amapiano/Extra/Kweyama Brothers, Mpura, Abidoza, Thabiso Lavish - Impilo yaseSandton.mp3"
    y, sr = librosa.load(mp3)

    yt = YouTube("https://www.youtube.com/watch?v=GdaeHPl6DRc")
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_bytes=BytesIO()
    audio_stream.stream_to_buffer(audio_bytes)
    
    classify(y, sr)

def download_audio_from_youtube(url, output="temp.mp3"):
    """
    Download audio from a YouTube URL and save it as an MP3 file.

    Args:
        url (str): YouTube URL of the audio source.
        output_dir (str): Directory to save the downloaded MP3 file.

    Returns:
        output_path (str): Path to the downloaded MP3 file.
    """
    print("Downloading song")
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output)
    return output

def load_audio_with_librosa(audio_path):
    """
    Load audio from an MP3 file using Librosa.

    Args:
        audio_path (str): Path to the MP3 audio file.

    Returns:
        y (np.ndarray): Audio data as a numpy array.
        sr (int): Sample rate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr  

def main():
    print("Choose an option:")
    print("1. Load audio from a directory")
    print("2. Load audio from a YouTube URL")

    option = input("Enter the option (1/2): ")

    if option == "1":
        audio_path = input("Enter the directory path containing the audio file: ")
    elif option == "2":
        # Ask for a URL input from the user
        url = input("Enter the YouTube URL of the audio source: ")

        # Download the audio as an MP3 file
        audio_path = download_audio_from_youtube(url)
    else:
        print("Invalid option. Please choose 1 or 2.")
        return

    # Load the selected audio with Librosa
    y, sr = load_audio_with_librosa(audio_path)

    # Call the classification function
    classify(y, sr)

if __name__ == "__main__":
    main()



    


    
