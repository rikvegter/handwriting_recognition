#!/bin/bash

# Whether to enable (0) or disable (non-0) debug logging.
DEBUG_LOGGING=1

# The names of Hebrew letters in the order as used in our project.
LETTER_NAMES=(Alef Bet Gimel Dalet He Waw Zayin Het Tet Yod Kaf Kaf-final Lamed Mem-medial Mem Nun-medial Nun-final Samekh Ayin Pe Pe-final Tsadi-medial Tsadi-final Qof Resh Shin Taw)

function get_contents() {
    # Retrieve the contents of a docx file.
    #
    # * param1: The path to the docs file.

    unzip -p "$1" word/document.xml
}

function get_ascii_dec() {
    # Gets the decimal ascii value of a single character.
    #
    # * param1: The character whose ascii value to retrieve.

    LC_CTYPE=C printf '%d' "'$1"
}

function get_name_from_idx() {
    # Gets the letter associated with the given index.
    # If the index is out of bounds, the original index is returned.
    #
    # * param1: The index for which to look up the associated letter.

    local idx=$1
    if ((idx >= 0 && idx < 27)); then
        echo ${LETTER_NAMES[$1]}
    else
        echo $1
    fi
}

function map_to_our_idx() {
    # Sadly, our labels are not quite the same as the official ones... :(
    # This method maps the indices based on the official unicode values
    # to our indices.
    #
    # * param1: The unicode-based index (i.e. range [0 26]) of a character.

    # Swap Kaf and Kaf-final
    if [ $1 -eq 10 ]; then echo 11;
    elif [ $1 -eq 11 ]; then echo 10;

    # Swap Mem and Mem-final
    elif [ $1 -eq 13 ]; then echo 14;
    elif [ $1 -eq 14 ]; then echo 13;

    # Swap Nun-final and Nun-medial
    elif [ $1 -eq 15 ]; then echo 16;
    elif [ $1 -eq 16 ]; then echo 15;

    # Swap Pe and Pe-final
    elif [ $1 -eq 19 ]; then echo 20;
    elif [ $1 -eq 20 ]; then echo 19;

    # Swap Tsadi-medial and Tsadi-final
    elif [ $1 -eq 21 ]; then echo 22;
    elif [ $1 -eq 22 ]; then echo 21;

    # The dots above the letters are ignored
    elif [ $1 -eq -33 ]; then echo "";
    elif [ $1 -eq -713 ]; then echo "";

    # Ignore the left-to-right mark
    elif [ $1 -eq 6718 ]; then echo "";
    # Ignore the right-to-left mark as well
    elif [ $1 -eq 6719 ]; then echo "";

    else
        echo $1
    fi
}

function get_hebrew_idx() {
    # Gets the unicode-based index (set to range [0 26]) of a hebrew character.
    #
    # * param1: The character for which to look up the unicode-based index.

    local ascii=$(get_ascii_dec "$1")
    real_idx=$(($ascii - 1488))
    map_to_our_idx $real_idx
}

function decode_hebrew() {
    # Decodes hebrew text into indices as used by our project.
    # Each line of the output contains a word built from space-separated index values.
    #
    # * param1: The hebrew text to decode
    # * param2: The output file.

    echo "$1" |  while read -r line; do
        # Use sed to place spaces between every character to make processing it easier. Also, remove blank lines.
        chars=$(echo "$line" | sed -e 's/\(.\)/\1\n/g' | sed '/^$/d' | while read -r char; do get_hebrew_idx "$char"; done | paste -sd' ' | sed "s/ \+/ /g;s/^ //g;s/ $//g")
        # Invert the order so we end up with the correct order of the output
        chars=$(echo "$chars" | awk -F" " '{for (i=NF; i; i--) printf "%s ",$i; print ""}')

        # Don't write empty lines to the output file
        test -z "$chars" || echo "$chars" >> "$2"

        if [ $DEBUG_LOGGING -eq 0 ]; then
            read -ra indices <<< "$chars"
            names=$(for idx in "${indices[@]}"; do get_name_from_idx $idx; done | paste -sd' ')
            echo "$line"
            echo "$chars"
            echo "$names"
            echo "============="
            echo ""
         fi
    done
}

find . -type f | grep "characters\.docx$" | while read -r line; do
    hebrew=$(get_contents "$line" | grep -oP "[^\x00-\x7F]*")
    file_out="$line.txt"
    # Clear the file
    > "$file_out"
    decode_hebrew "$hebrew" "$file_out"
done

