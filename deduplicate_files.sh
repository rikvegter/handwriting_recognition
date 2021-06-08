#!/bin/bash

DATA_FOLDER="dataset"

# Whether to enable (0) or disable (non-0) debug logging.
DEBUG_LOGGING=1

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
        local unique_files_lookup="${unique_files[$identifier]}"
        test -z $unique_files_lookup && continue

        if [[ "${unique_files[$identifier]}" -eq 1 ]]; then
            unique_file="$potentially_unique_file_line"
            break
        fi
    done < <(find "$DATA_FOLDER/$bucket/$char" -type f )
}

deduplicate_files() {
    local char=$1
    local duplicates="$2"

    local bucket=$(echo "$duplicates" | grep -Po "(?<=$DATA_FOLDER\/)[0-9]*(?=\/)" | head -n1)

    while read -r duplicate_file_line; do
        find_unique_file $char $bucket
        test -z "$unique_file" && { echo "Failed to find unique file in dir: \"$DATA_FOLDER/$bucket/$char\" to replace with duplicate file: \"$duplicate_file_line\""; exit; }

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
    done < <(find data/$char -type f | grep -o "[0-9]*-line-[0-9]*" | sort | uniq -c | grep "  1 " | awk '{print $2}')

    while read -r file_to_analyze; do
        local duplicates=$(find $DATA_FOLDER -type f | grep -v "_augmented" | grep $char | grep "$file_to_analyze" | paste -sd' ')
        local in_bucket_count=$(echo "$duplicates" | grep -Po "(?<=$DATA_FOLDER\/)[0-9]*(?=\/)" | sort | uniq -c | wc -l)
        test $in_bucket_count -gt 1 && { log ""; log ""; log "Leaked duplicates: "; leaked_dupes=$(echo "$duplicates" | tr " " "\n"); log "$leaked_dupes"; }
        test $in_bucket_count -gt 1 && deduplicate_files $char "$duplicates"
    done < <(find data/$char -type f | grep -o "[0-9]*-line-[0-9]*" | sort | uniq -c | grep -v "  1 " | awk '{print $2}')
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
done < <(find data -maxdepth 1 -type d -printf '%f\n' | tail -n+2)

echo "All buckets have been deduplicated!"

