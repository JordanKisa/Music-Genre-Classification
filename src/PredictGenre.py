##


import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array

model = load_model('./src/final_model.keras')


scaler = joblib.load('./src/minmax_scaler.pkl')

mp3 = "data/Amapiano/Extra/Kweyama Brothers, Mpura, Abidoza, Thabiso Lavish - Impilo yaseSandton.mp3"


y, sr = librosa.load(mp3)
parent_folder = os.path.basename(os.path.dirname(os.path.dirname(mp3)))

# Calculate tempo
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
duration = librosa.get_duration(y=y, sr=sr)
        
# Calculate MFCCs and mean and Variance
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfccs, axis=1)
mfcc_variance = np.var(mfccs, axis=1)

# Calculate features
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
row = [tempo, chroma_mean, chroma_var, spectral_centroid_mean,  spectral_centroid_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,  onset_strength_mean, onset_strength_var,
        harmonic_mean, harmonic_var, percussive_mean, percussive_var, tonnetz_mean, tonnetz_var, rms_mean, rms_var]
row.extend(list(mfcc_mean))
row.extend(list(mfcc_variance))

row = np.expand_dims(row, axis=0)
row = scaler.transform(row)
np.set_printoptions(threshold=np.inf)


segment_duration = 3
n_frames_per_segment = int(segment_duration * sr) 
num_segments = int(np.floor(len(y) / n_frames_per_segment))


label_mapping = {'Funky-Soul': 0, 'Favela-Funk': 1, 'Samba': 2,
                  'Amapiano': 3, 'Jungle': 4, 'Reggae': 5, 
                  'Soca': 6, 'Bossa-Nova': 7, 'RnB': 8,
                    'Neo-Soul': 9, 'Reggaeton': 10}
index_to_label = {index: label for label, index in label_mapping.items()}

data = []

# Loop through segments
for i in range(num_segments):
    # Calculate the start and end frames for the current segment
    start_frame = i * n_frames_per_segment
    end_frame = min((i + 1) * n_frames_per_segment, len(y))
    
    # Extract the segment of audio
    segment = y[start_frame:end_frame]
    
    mel = librosa.feature.melspectrogram(y=segment, sr=sr)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    plt.imsave("temp.png", mel_db, cmap='Greys')

    image = cv2.imread("temp.png", 0)
    image = cv2.resize(image, (131,128))
    image_arr = img_to_array(image)
    image_arr = image_arr / 255.0
    data.append(image_arr)

    

predictions = model.predict([np.array(data), np.tile(row, (num_segments, 1))])
predicted_labels = np.argmax(predictions, axis=1)
pred_labels_class = [index_to_label[label] for label in predicted_labels]


# Calculate the total count of values
total_count = len(pred_labels_class)

# Calculate the count of each unique value
unique_values = set(pred_labels_class)

count_dict = {value: pred_labels_class.count(value) for value in unique_values}
sorted_values = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

for value, count in sorted_values:
    percentage = (count / total_count) * 100
    print(f"Genre: {value} Likelihood {percentage:.2f}%")

    