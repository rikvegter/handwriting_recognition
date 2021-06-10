#!/bin/bash

ARCHIVE="$1/monkbrill.tar.gz"
OUTPUT_DIR="$1/data/original"

DEBUG_LOGGING=$2


function unpack() {
    mkdir -p "$OUTPUT_DIR"
    tar -xzf "$ARCHIVE" -C "$OUTPUT_DIR"
}

if [ -d "$OUTPUT_DIR" ]; then 
    echo "Data directory exist already! Nothing left to do!"
elif [ -f "$ARCHIVE" ]; then
    unpack
else
    echo "Failed to extract archive \"$ARCHIVE\": File not found!"
    exit 1
fi 





