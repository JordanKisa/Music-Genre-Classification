import concurrent.futures
import librosa
import os
import csv
import sys
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm  # Import tqdm for tracking progress
import warnings

warnings.filterwarnings("ignore") #librosa has some and the libraries it uses will be depreciating the use of a library for loading. Just a future warning

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(audio_file)))

# Calculate tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
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
    row = [os.path.basename(audio_file), tempo, chroma_mean, chroma_var, spectral_centroid_mean,  spectral_centroid_var,
            zero_crossing_rate_mean, zero_crossing_rate_var,  onset_strength_mean, onset_strength_var,
            harmonic_mean, harmonic_var, percussive_mean, percussive_var, tonnetz_mean, tonnetz_var, rms_mean, rms_var]
    row.extend(list(mfcc_mean))
    row.extend(list(mfcc_variance))
    row.append(parent_folder)

    return row

def process_audio_files(audio_directory):
    audio_files = [os.path.join(audio_directory, file) for file in os.listdir(audio_directory)]
    total_files = len(audio_files)

     # Get the parent folder name
    parent_folder = os.path.basename(os.path.dirname(audio_directory))

    # Create a CSV filename based on the parent folder and current folder
    csv_filename = f"{parent_folder}_{os.path.basename(audio_directory)}_features_detailed.csv"
    parent_directory = os.path.dirname(audio_directory)
    csv_filepath = os.path.join(parent_directory, csv_filename)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_features, audio_files), total=total_files, desc="Processing"))

    with open(csv_filepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write header row
        header = ["File", "Tempo"]
        feature_names = ["Chroma", "SpectralCentroid", "ZeroCrossingRate", "OnsetStrength", "Harmonic", "Percussive", "Tonnetz", "RMS"]
        for name in feature_names:
            header.extend([f"{name}_mean", f"{name}_var"])
        header += [f"MFCC{i+1}_mean" for i in range(13)] + [f"MFCC{i+1}_variance" for i in range(13)] + ["Label"]
        csvwriter.writerow(header)

        for result in results:
            csvwriter.writerow(result)

    print(f"Extracted features saved to {csv_filepath}")
                
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DataExtractorCSV.py audio_directory csv")
        sys.exit(1)
    
    audio_directory = sys.argv[1]
    process_audio_files(audio_directory)
