import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for tracking progress

import warnings
warnings.filterwarnings("ignore")


def generate_and_save_spectrograms(music_directory, spectrogram_directory):
    # Create the spectrogram directory if it doesn't exist
    if not os.path.exists(spectrogram_directory):
        os.makedirs(spectrogram_directory)

    # Loop through music files
    for filename in tqdm(os.listdir(music_directory)):
        if filename.endswith(".mp3"):
            file_path = os.path.join(music_directory, filename)

            # Load the audio file
            y, sr = librosa.load(file_path)

            # Compute the spectrogram
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            beats = librosa.util.fix_frames(beats, x_min=0)
            chroma_sync = librosa.util.sync(chroma, beats)

            chroma_lag = librosa.feature.stack_memory(
                chroma_sync, n_steps=2, mode="edge"
            )
            split_index = chroma_lag.shape[0] // 2
            temporal_chroma = chroma_lag[:split_index] + chroma_lag[split_index:]

            # Save the spectrogram as an image
            spectrogram_image_path = os.path.join(spectrogram_directory, f'{os.path.splitext(filename)[0]}.png')
            plt.imsave(spectrogram_image_path, temporal_chroma)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python generate_spectrograms.py music_directory spectrogram_directory"
        )
        sys.exit(1)

    music_files_directory = sys.argv[1]
    spectrogram_images_directory = sys.argv[2]

    generate_and_save_spectrograms(music_files_directory, spectrogram_images_directory)
