#!/bin/bash

ORIGINAL_DATA_FOLDER="$1/data/original"
DATASET_FOLDER="$1/data/dataset"

DEBUG_LOGGING=$2


function log() {
    # Logs a message, if DEBUG_LOGGING is enabled.
    #
    # * param1: The message to log.
    test $DEBUG_LOGGING -eq 0 && echo "$1"
}

swap_files() {
    log "Swap $1 $2"

    dir_1="$(dirname "${1}")"
    dir_2="$(dirname "${2}")"

    mv "$1" "$dir_2"
    mv "$2" "$dir_1"
}

get_identifier() {
    echo $(grep -o "[0-9]*-line-[0-9]*" <<< "$1")
}

find_unique_file() {
    local char=$1
    local bucket=$2

    unset unique_file
    while read -r potentially_unique_file_line; do
        local identifier=$(get_identifier "$potentially_unique_file_line")

        # Kaf-final has 2 images that do not follow the same naming scheme. However, it has been
        # manually verified that these two images are unique within kaf-final.
        test -z $identifier && { unique_file="$potentially_unique_file_line"; continue; }

        if [[ "${unique_files[$identifier]}" -eq 1 ]]; then
            unique_file="$potentially_unique_file_line"
            break
        fi
    done < <(find "$DATASET_FOLDER/$bucket/$char" -type f )
}

deduplicate_files() {
    local char=$1
    local duplicates="$2"

    local bucket=$(echo "$duplicates" | grep -Po "(?<=$DATASET_FOLDER\/)[0-9]*(?=\/)" | head -n1)

    while read -r duplicate_file_line; do
        find_unique_file $char $bucket
        test -z "$unique_file" && { echo "Failed to find unique file in dir: \"$DATASET_FOLDER/$bucket/$char\" to replace with duplicate file: \"$duplicate_file_line\""; exit 1; }

        swap_files "$unique_file" "$duplicate_file_line"

        unique_file_identifier=$(get_identifier "$unique_file")
        unique_files[$unique_file_identifier]=0
    done < <(echo "$duplicates" | tr " " "\n" | grep -v "\/$bucket\/")
}

get_split_duplicates() {
    local char="$1"

    unset unique_files
    declare -gA unique_files
    while read -r unique_file_line; do
        unique_files[$unique_file_line]=1
    done < <(find "$ORIGINAL_DATA_FOLDER/$char" -type f | grep -o "[0-9]*-line-[0-9]*" | sort | uniq -c | grep "  1 " | awk '{print $2}')

    while read -r file_to_analyze; do
        local duplicates=$(find $DATASET_FOLDER -type f | grep -v "_augmented" | grep $char | grep "$file_to_analyze" | paste -sd' ')
        local in_bucket_count=$(echo "$duplicates" | grep -Po "(?<=$DATASET_FOLDER\/)[0-9]*(?=\/)" | sort | uniq -c | wc -l)
        test $in_bucket_count -gt 1 && { log ""; log ""; log "Leaked duplicates: "; leaked_dupes=$(echo "$duplicates" | tr " " "\n"); log "$leaked_dupes"; }
        test $in_bucket_count -gt 1 && deduplicate_files $char "$duplicates"
    done < <(find "$ORIGINAL_DATA_FOLDER/$char" -type f | grep -o "[0-9]*-line-[0-9]*" | sort | uniq -c | grep -v "  1 " | awk '{print $2}')
}

while read -r character; do
    width=42
    offset=3
    header_width=$((width + 2 * offset + 2))
    printf "\n\n%${header_width}s\n" | tr " " "-"
    printf "%${offset}s|%${width}s|%${offset}s\n"
    printf "%${offset}s/%${width}s\%${offset}s\n"
    printf "  >               %-14s               < \n" $character
    printf "%${offset}s\%${width}s/%${offset}s\n"
    printf "%${offset}s|%${width}s|%${offset}s\n"
    printf "%${header_width}s\n" | tr " " "-"
    get_split_duplicates "$character"
done < <(find "$ORIGINAL_DATA_FOLDER" -maxdepth 1 -type d -printf '%f\n' | tail -n+2)

echo "All buckets have been deduplicated!"

