#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 input_directory output_directory crop_duration"
    exit 1
fi

input_directory="$1"
output_directory="$2"
crop_duration="$3"

mkdir -p "$output_directory"

for file in "$input_directory"/*.mp3; do
    filename=$(basename "$file")
    output_file="$output_directory/$filename"

    ffmpeg -i "$file" -t "$crop_duration" -acodec copy "$output_file"
done