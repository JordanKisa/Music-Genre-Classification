import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for tracking progress

def generate_and_save_spectrograms(music_directory, spectrogram_directory):
    # Create the spectrogram directory if it doesn't exist
    if not os.path.exists(spectrogram_directory):
        os.makedirs(spectrogram_directory)

    # Loop through music files
    for filename in tqdm(os.listdir(music_directory)):
        if filename.endswith('.mp3'):
            file_path = os.path.join(music_directory, filename)
            
            # Load the audio file
            y, sr = librosa.load(file_path)
            
            # Compute the spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectogram_db = librosa.amplitude_to_db(spectrogram)
            
            # Save the spectrogram as an image
            spectrogram_image_path = os.path.join(spectrogram_directory, f'{os.path.splitext(filename)[0]}.png')
            plt.imsave(spectrogram_image_path, spectogram_db, cmap='cool')

    print('Spectrogram generation and saving completed.')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_spectrograms.py music_directory spectrogram_directory")
        sys.exit(1)

    music_files_directory = sys.argv[1]
    spectrogram_images_directory = sys.argv[2]

    generate_and_save_spectrograms(music_files_directory, spectrogram_images_directory)
