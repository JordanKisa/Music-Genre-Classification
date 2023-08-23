import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures  # Import concurrent.futures for concurrency

import warnings
warnings.filterwarnings("ignore")

def generate_mel_spectrogram(music_directory,filename):
    file_path = os.path.join(music_directory, filename)
    
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram).tobytes()
    
    return os.path.splitext(filename)[0], mel_spectrogram_db

def generate_and_save_mel_spectrograms(music_directory, csv_filename):
    mel_spectrograms = []
    titles = []
    total = len(os.listdir(music_directory))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for filename, mel_spectrogram in tqdm(executor.map(generate_mel_spectrogram, [music_directory] * total, os.listdir(music_directory)), total=total):
            mel_spectrograms.append(mel_spectrogram)  # Flatten the mel spectrogram
            titles.append(filename)  # Use the filename as the label

    # Create a DataFrame to store the flattened mel spectrograms and labels
    mel_data = pd.DataFrame({'title': titles,  'spectrogram': mel_spectrograms, 'label': [os.path.basename(os.path.dirname(music_directory))] * total})

    # Save DataFrame to a CSV file
    mel_data.to_csv(csv_filename, index=False)

    print('Mel spectrogram generation and saving completed.')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_mel_spectrograms.py music_directory output_csv_filename")
        sys.exit(1)

    music_files_directory = sys.argv[1]
    csv_output_filename = sys.argv[2]

    generate_and_save_mel_spectrograms(music_files_directory, csv_output_filename)
