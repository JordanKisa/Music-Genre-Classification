import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for concurrency

import warnings
warnings.filterwarnings("ignore")

def process_audio_file(file_path):
    y, sr = librosa.load(file_path)

    #chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    mel_delta2 = librosa.feature.delta(mel_db, order=2)

            
    #tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    #beats = librosa.util.fix_frames(beats, x_min=0)
    #chroma_sync = librosa.util.sync(chroma, beats)

    #chroma_lag = librosa.feature.stack_memory(
    #    chroma_sync, n_steps=2, mode="edge"
    #)
    #split_index = chroma_lag.shape[0] // 2
    #temporal_chroma = chroma_lag[:split_index] + chroma_lag[split_index:]

    return mel_db

def generate_and_save_spectrogram(filename, music_directory, spectrogram_directory):
    file_path = os.path.join(music_directory, filename)
    temporal_chroma = process_audio_file(file_path)

    spectrogram_image_path = os.path.join(spectrogram_directory, f'{os.path.splitext(filename)[0]}.png')
    plt.imsave(spectrogram_image_path, temporal_chroma, cmap='Greys')

def generate_and_save_spectrograms_concurrent(music_directory, spectrogram_directory):
    if not os.path.exists(spectrogram_directory):
        os.makedirs(spectrogram_directory)

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in os.listdir(music_directory):
            if filename.endswith(".mp3"):
                futures.append(
                    executor.submit(generate_and_save_spectrogram, filename, music_directory, spectrogram_directory)
                )

        for future in tqdm(futures, total=len(futures), desc="Generating Spectrograms", smoothing=0):
            future.result()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python generate_spectrograms.py music_directory spectrogram_directory"
        )
        sys.exit(1)

    music_files_directory = sys.argv[1]
    spectrogram_images_directory = sys.argv[2]

    generate_and_save_spectrograms_concurrent(music_files_directory, spectrogram_images_directory)
