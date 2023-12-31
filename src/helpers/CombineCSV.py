import os
import pandas as pd

# Define the root directory to search for CSV files
root_directory = './data/'

print(os.path.isdir(root_directory))

# Define the string that the subdirectories should contain
target_substring = 'spectrogram'

# Initialize an empty DataFrame to store combined data
frames = []

# Recursively search for CSV files in subdirectories
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.csv') and target_substring in file:
            file_path = os.path.join(root, file)
            print(f"found {file_path}")
            df = pd.read_csv(file_path)
            
            ##Uncomment if you need to label data
            #parent_folder = os.path.basename(root)
            #df['label'] = parent_folder
            frames.append(df)

combined_data = pd.concat(frames)
# Define the path to save the combined CSV file
output_file_path = './data/spectrograms.csv'

# Save the combined DataFrame to a CSV file
combined_data.to_csv(output_file_path, index=False)

print("CSV files combined and saved to:", output_file_path)