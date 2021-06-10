#!/bin/bash



# This script uses ImageMorph to apply data augmentation on a set of images.
# Because ImageMorph does not include a license, it is not provided with
# this script. Instead, this script will download and compile it before use.
# Link to ImageMorph: https://github.com/GrHound/imagemorph.c



DATASET_DIR="$1/data/dataset"

IMAGEMORPH_DIR="$1/ImageMorph"
IMAGE_MORPH_SOURCE="imagemorph.c"
IMAGE_MORPH_SOURCE_URL="https://raw.githubusercontent.com/GrHound/imagemorph.c/master/imagemorph.c"
IMAGE_MORPH="$IMAGEMORPH_DIR/imagemorph"
AUGMENTED_SUFFIX="_augmented"

TMP_DIR="$1/tmp"

SETTING_IMAGE_MORPH_DISPLACEMENT=1.4
SETTING_IMAGE_MORPH_SMOOTHING_RADIUS=14
SETTING_IMAGE_MORPH_COUNT=2

DEBUG_LOGGING=$2



function log() {
    # Logs a message, if DEBUG_LOGGING is enabled.
    #
    # * param1: The message to log.
    test $DEBUG_LOGGING -eq 0 && echo "$1"
}

function logged_cmd() {
    # Logs a command and then executes it. See #log().
    #
    # * param1: The command to log and execute.
    log "$1"; bash -c "$1"
}

function ensure_imagemorph_available() {
    log "imgmorph dir: $IMAGEMORPH_DIR"
    mkdir -p "$IMAGEMORPH_DIR"
    # Ensures that the image morph program is available.
    # If it isn't it will be downloaded (if needed) and compiled.
    test -f "$IMAGEMORPH_DIR/$IMAGE_MORPH_SOURCE" \
            || logged_cmd "wget -qO \"$IMAGEMORPH_DIR/$IMAGE_MORPH_SOURCE\" \"$IMAGE_MORPH_SOURCE_URL\"" \
            || { echo "Failed to download file from $IMAGE_MORPH_SOURCE_URL!" 1>&2; rm "$IMAGEMORPH_DIR/$IMAGE_MORPH_SOURCE"; exit 1; }

    test -f "$IMAGE_MORPH" \
            || logged_cmd "gcc \"$IMAGEMORPH_DIR/$IMAGE_MORPH_SOURCE\" -static -static-libgcc -lm -o \"$IMAGE_MORPH\"" \
            || { echo "Failed to compile image morph!"; exit 2; }
}

function run_image_morph() {
    # Runs the image morph on a file.
    #
    # * param1: The path to the input file.
    # * param2: The path to the output file.
    # * param3: The displacement value to apply.
    # * param4: The smoothing radius to use.
    local file_in="$1"
    local file_out="$2"
    local displacement="$3"
    local smoothing_radius="$4"
    log "image_morph ($3, $4): $file_in -> $file_out"
    "$IMAGE_MORPH" "$displacement" "$smoothing_radius" < "$file_in" > "$file_out"
}

function convert_file() {
    # Converts a file from one format into another based on its extension.
    # ImageMagick's 'convert' utility is used for the convertion process.
    #
    # If the extension of the existing file is the same as the specified
    # extension, the file will not be converted, but simply copied to the
    # destination directory.
    #
    # * param1: The name (not path!) of the file to convert.
    # * param2: The directory the file is in.
    # * param3: The extension the output file should have (without dot!) E.g. "ppm".
    # * param4: The output directory of the file.
    local file_name="$1"
    local dir_in="$2"
    local new_extension="$3"
    local dir_out="$4"

    local file_name_base="${file_name%%.*}"
    local extension="${file_name##*.}"
    local converted_file="$dir_out/$file_name_base.$new_extension"
    test "$extension" = "$new_extension" && \
            logged_cmd "cp \"$dir_in/$file_name\" \"$converted_file\"" || \
            logged_cmd "convert \"$dir_in/$file_name\" \"$converted_file\""
}

function run_for_file() {
    # Runs ImageMorph for a given file.
    #
    # * param1: The full path to the file.
    # * param2: The directory to store the output files in.

    # The full path to the file. E.g. /my/path/to/my-image.ppm
    local file="$1"
    # The output directory.
    local dir_out="$2"

    log ""
    log "Running for file: $file. Output directory: $dir_out"

    # The file name + its extension. E.g. my-image.ppm
    local file_name=$(basename "$file")
    # The file's base name without extension. E.g. my-image
    local file_name_base="${file_name%%.*}"
    # The directory the file resides in. E.g. /my/path/to
    local file_dir=$(dirname "$file")

    # ImageMorph only works with ppm images, so convert the image to ppm first.
    convert_file "$file_name" "$file_dir" "ppm" "$TMP_DIR"

    local input_file="$TMP_DIR/$file_name_base.ppm"

    # The image morph program we use has a maximum value for the smoothing component,
    # As the smoothing value cannot be greater than min(width, height) / 2.5
    local ppmfile_dims=$(identify "$input_file" | awk '{print $3}')
    local width=$(grep -o "^[0-9]*" <<< "$ppmfile_dims")
    local height=$(grep -o "[0-9]*$" <<< "$ppmfile_dims")
    local min_dim=$(($width > $height ? $height : $width))
    local max_smoothing=$(awk "BEGIN {print int($min_dim / 2.5)}")
    local smoothing_radius=$(($SETTING_IMAGE_MORPH_SMOOTHING_RADIUS > $max_smoothing ? $max_smoothing : $SETTING_IMAGE_MORPH_SMOOTHING_RADIUS))

    for ((i=0; i<$SETTING_IMAGE_MORPH_COUNT; i++)); do
        local output_file_name="$file_name_base""_ImageMorph_$i.ppm"
        local output_file="$dir_out/$output_file_name"

        # Apply the image morph to the current iteration.
        run_image_morph "$input_file" "$output_file" $SETTING_IMAGE_MORPH_DISPLACEMENT $smoothing_radius
        # Convert the file back to pgm, so all files have the same format.
        convert_file "$output_file_name" "$dir_out" "pgm" "$dir_out"
        # Remove the old ppm file.
        logged_cmd "rm \"$output_file\""
    done

    # Clean up the temporary file.
    logged_cmd "rm \"$TMP_DIR/$file_name_base.ppm\""
}

function run_for_directory() {
    # Runs the data augmentation for all images in a given directory.
    #
    # * param1: The directory containing image files.

    # Remove trailing '/' if it exists.
    local dir_in="${1%/}"
    log "Processing directory: $dir_in"

    # Replace the part in the path (e.g. the '0' in 'dataset/0/Tsadi-final'
    # To the part + the suffix (e.g. to 'dataset/0_augmented/Tsadi-final').
    local dir_out=$(perl -Xpe "s/(?<=\/[0-9]{1,3})(?=\/)/$AUGMENTED_SUFFIX/" <<< "$dir_in") || \
            { echo "Failed to use find dir_out in dir_in: \"$dir_in\". Is your Perl up-to-date?"; exit 11; }

    mkdir -p "$dir_out" || { echo "Failed to create directory: \"$dir_out\""; exit 12; }

    find "$dir_in" -type f -printf "%f\n" | while read -r line; do
        run_for_file "$dir_in/$line" "$dir_out"
    done
}

function run() {
    local dir_count=$(find "$DATASET_DIR" -type d -links 2 ! -empty | grep -v "$AUGMENTED_SUFFIX" | wc -l)
    local processed=0
    find "$DATASET_DIR" -type d -links 2 ! -empty | grep -v "$AUGMENTED_SUFFIX" | while read -r line; do
        test $(($processed % 5)) -eq 0 && printf "Processed %3d / %3d directories!\n" $processed $dir_count
        run_for_directory "$line"
        processed=$(($processed + 1))
    done
}

# Create the directories for the ImageMorph utility and the temporary files.
mkdir -p "$IMAGEMORPH_DIR"
mkdir -p "$TMP_DIR"

# Ensure the ImageMorph is available (i.e. has been downloaded and compiled).
ensure_imagemorph_available || { echo "Failed to obtain ImageMorph!"; exit 111; }
# Run data augmentation.
run || { echo "Failed to apply data augmentation!"; exit 112; }

# Remove the temporary directory if it's empty.
test -z $(ls -A "$TMP_DIR") && rm -d "$TMP_DIR"

