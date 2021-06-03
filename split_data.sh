#!/bin/bash

INPUT="data"
OUTPUT="dataset"
PARTS=5

function split_files() {
    split=$1
    total_count=$2
    name="$3"
    file_list="$4"
    
    dir="$OUTPUT/$split/$name/"
 
    mkdir -p "$dir"
    
    # Not every letter will have a perfectly divisible number of files.
    # So just try to divide them as evenly as possible.
    skip=$(awk "BEGIN {print int(1 + $split * $total_count / $PARTS)}")
    # When we're at the last split, just grab all remaining files.
    test $split -eq $(($PARTS - 1)) && grab=99999 || grab=$(awk "BEGIN {print int(($split + 1) * $total_count / $PARTS) - $skip + 1}")
 
    echo "$file_list" | tail -n+$skip | head -n$grab | xargs -d'\n' -I{} cp -u {} "$dir"
}

find "$INPUT" -maxdepth 1 -type d | tail -n+2 | while read -r directory; do
    files=$(find "$directory" -type f | shuf)
    character_name=$(echo "$directory" | grep -o "[a-zA-Z-]*$")
    file_count=$(echo "$files" | wc -l)

    for ((i=0; i<$PARTS; i++)); do
        split_files $i $file_count "$character_name" "$files"
    done
done






