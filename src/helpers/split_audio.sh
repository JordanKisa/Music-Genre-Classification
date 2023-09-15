#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 input_directory output_directory"
    exit 1
fi

input_directory="$1"
output_directory="$2"

mkdir -p "$output_directory"

for file in "$input_directory"/*.mp3; do
    filename=$(basename "$file")
    
    ffmpeg -i "$file" -f segment -segment_time 3 -c copy "$output_directory/${filename%.mp3}_%03d.mp3"
done
