import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for tracking progress

def generate_and_save_mel_spectrograms(music_directory, csv_filename):
    mel_spectrograms = []
    labels = []

    # Loop through music files
    for filename in tqdm(os.listdir(music_directory)):
        if filename.endswith('.mp3'):
            file_path = os.path.join(music_directory, filename)
            
            # Load the audio file
            y, sr = librosa.load(file_path)
            
            # Compute the mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram)
            
            mel_spectrograms.append(mel_spectrogram_db.flatten())  # Flatten the mel spectrogram
            labels.append(os.path.splitext(filename)[0])  # Use the filename as the label

    # Create a DataFrame to store the flattened mel spectrograms and labels
    mel_data = pd.DataFrame({'mel_spectrogram': mel_spectrograms, 'label': labels})

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
