#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 source_directory duration_threshold"
    exit 1
fi

source_directory="$1"
duration_threshold="$2"

for file in "$source_directory"/*.mp3; do
    duration=$(ffprobe -i "$file" -show_entries format=duration -v quiet -of csv="p=0")
    if (( $(bc <<< "$duration < $duration_threshold") )); then
        echo "File: $(basename "$file") | Duration: $duration seconds"
    fi
done
