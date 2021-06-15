#!/bin/bash

# Whether to enable (0) or disable (non-0) debug logging.
DEBUG_LOGGING=1

INPUT_FILE="ngrams_frequencies_withNames.xlsx"
OUTPUT_FILE="ngrams_frequencies_withNames.xlsx.txt"

# The names of Hebrew letters in the order as used in our project.
LETTER_NAMES=(Alef Bet Gimel Dalet He Waw Zayin Het Tet Yod Kaf Kaf-final Lamed Mem-medial Mem Nun-medial Nun-final Samekh Ayin Pe Pe-final Tsadi-medial Tsadi-final Qof Resh Shin Taw)

function log() {
    # Logs a message, if DEBUG_LOGGING is enabled.
    #
    # * param1: The message to log.
    test $DEBUG_LOGGING -eq 0 && echo "$1"
}

function get_character_index() {
    local character_name=$(echo "$1" | sed "s/Tsadi/Tsadi-medial/g;s/Tasdi-final/Tsadi-final/g")
    local idx="${LETTER_MAP[$character_name]}"
    test -z "$idx" && { echo "Failed to find index for character \"$1\"!"; exit 1; }
    echo "$idx"
}

function get_character_indices() {
    read -ra letter_names <<< $(echo "$1" | sed "s/_/ /g")
    for letter_name in "${letter_names[@]}"; do get_character_index "$letter_name"; done | paste -sd' '
}

function print_letter_map() {
    local len=${#LETTER_MAP[@]}
    for (( idx = 0; idx < $len; ++idx )); do
        local letter_name=${LETTER_NAMES[$idx]}
        echo "$letter_name ${LETTER_MAP[$letter_name]}"
    done
    echo ""
}

function make_letter_map() {
    # Store all the letters in an associative array for easier index lookups.
    declare -gA LETTER_MAP
    len_labels=${#LETTER_NAMES[@]}
    for (( idx = 0; idx < $len_labels; ++idx )); do
        local letter_name=${LETTER_NAMES[$idx]}
        LETTER_MAP[$letter_name]=$idx
    done
    test $DEBUG_LOGGING -eq 0 && print_letter_map
}

make_letter_map

mapfile -t combinations < <(unzip -p "$INPUT_FILE" "xl/sharedStrings.xml" | grep -oP "(?<=<t>)[a-zA-Z_-]*" | tail -n+4)
mapfile -t frequencies< <(unzip -p "$INPUT_FILE" "xl/worksheets/sheet1.xml" | grep -oP "(?<=<v>)[0-9]*(?=\.0)")

len=${#combinations[@]}
len2=${#frequencies[@]}
test $len -ne $len2 && { echo "Number of character combinations ($len) does not match the number of frequencies ($len2)!"; exit 1; }

# Clear the file
> "$OUTPUT_FILE"

for (( idx = 0; idx < $len; ++idx )); do
    printf '%d %s\n' ${frequencies[$idx]} "$(get_character_indices ${combinations[$idx]})" >> "$OUTPUT_FILE"
done

