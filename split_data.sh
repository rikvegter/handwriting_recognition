#!/bin/bash

INPUT="data"
OUTPUT="dataset"
PARTS=5

function split_files() {
    split=$1
    total_count=$2
    name="$3"
    
    dir="$OUTPUT/$split/$name/"
 
    mkdir -p "$dir"
    
    # Not every letter will have a perfectly divisible number of files.
    # So just try to divide them as evenly as possible.
    skip=$(awk "BEGIN {print int(1 + $split * $total_count / $PARTS)}")
    # When we're at the last split, just grab all remaining files.
    test $split -eq $(($PARTS - 1)) && grab=99999 || grab=$(awk "BEGIN {print int(($split + 1) * $total_count / $PARTS) - $skip + 1}")
 
    find "$INPUT/$name" -type f | tail -n+$skip | head -n$grab | xargs -d'\n' -I{} cp -u {} "$dir"
}

declare -A file_counts
while read -r line; do
    name=$(echo "$line" | grep -o "[a-zA-Z-]*$")
    count=$(find "$line" -type f | wc -l)
    file_counts[$name]=$(($count))
done < <(find "$INPUT" -maxdepth 1 -type d | tail -n+2)

for entry in "${!file_counts[@]}"; do
    for ((i=0; i<$PARTS; i++)); do
        split_files $i $((file_counts["$entry"])) "$entry"
    done
done




