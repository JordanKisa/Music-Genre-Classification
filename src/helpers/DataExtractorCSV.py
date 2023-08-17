import librosa
import os
import csv
import sys
import numpy as np

def extract_features(audio_directory):
    # Get the parent folder name
    parent_folder = os.path.basename(os.path.dirname(audio_directory))

    # Create a CSV filename based on the parent folder and current folder
    csv_filename = f"{parent_folder}_{os.path.basename(audio_directory)}_features.csv"
    parent_directory = os.path.dirname(audio_directory)
    csv_filepath = os.path.join(parent_directory, csv_filename)

    with open(csv_filepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header row
        header = ["File", "Tempo",] + [f"MFCC{i+1}_mean" for i in range(20)] + [f"MFCC{i+1}_variance" for i in range(20)]
        csvwriter.writerow(header)
        
        # Loop over all files in the directory
        for filename in os.listdir(audio_directory):
            audio_path = os.path.join(audio_directory, filename)
            
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            # Calculate tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Calculate MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
             # Calculate mean and variance of each MFCC coefficient
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_variance = np.var(mfccs, axis=1)
            
            # Combine all the features
            row = [filename, tempo] + list(mfcc_mean) + list(mfcc_variance)
            csvwriter.writerow(row)
                
    print(f"Extracted features saved to {csv_filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DataExtractorCSV.py audio_directory")
        sys.exit(1)
    
    audio_directory = sys.argv[1]
    extract_features(audio_directory)
