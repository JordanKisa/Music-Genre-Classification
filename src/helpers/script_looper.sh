#!/bin/bash

# List of directories and corresponding output filenames
directories=("Amapiano" "Bossa-Nova" "Favela-Funk" "Funky-Soul" "Jungle" "Neo-Soul" "Reggae" "Reggaeton" "RnB" "Samba" "Soca")

# Loop over directories and output filenames
for ((i=0; i<${#directories[@]}; i++)); do
    echo "Starting Spectrogram Extraction for ${directories[i]}"
    python3 src/helpers/SpectogramImageExtractor.py "data/${directories[i]}/3-seconds" "data/${directories[i]}/${directories[i]}"
    echo "Spectrogram extraction for ${directories[i]} completed."
done