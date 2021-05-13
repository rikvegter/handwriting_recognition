#!/bin/bash

DATASET_DIR="dataset"

DATA_AUGMENTATION_DIR="DataAugmentation"
IMAGE_MORPH_SOURCE="imagemorph.c"
IMAGE_MORPH_SOURCE_URL="https://raw.githubusercontent.com/GrHound/imagemorph.c/master/imagemorph.c"
IMAGE_MORPH="$DATA_AUGMENTATION_DIR/imagemorph"
AUGMENTED_SUFFIX="_augmented"

TMP_DIR="tmp"

SETTING_IMAGE_MORPH_DISPLACEMENT=1.4
SETTING_IMAGE_MORPH_SMOOTHING_RADIUS=14
SETTING_IMAGE_MORPH_COUNT=30


function ensure_imagemorph_available() {
    test -f "$DATA_AUGMENTATION_DIR/$IMAGE_MORPH_SOURCE" \
            || wget -qO "$DATA_AUGMENTATION_DIR/$IMAGE_MORPH_SOURCE" "$IMAGE_MORPH_SOURCE_URL" \
            || { echo "Failed to download file from $IMAGE_MORPH_SOURCE_URL!" 1>&2; rm "$DATA_AUGMENTATION_DIR/$IMAGE_MORPH_SOURCE"; exit 1; }

    test -f "$IMAGE_MORPH" \
            || gcc "$DATA_AUGMENTATION_DIR/$IMAGE_MORPH_SOURCE" -static -static-libgcc -lm -o "$IMAGE_MORPH" \
            || { echo "Failed to compile image morph!"; exit 2; }
}


function run_image_morph() {
    filein="$1"
    fileout="$2"
    displacement="$3"
    smoothing_radius="$4"

    ./"$IMAGE_MORPH" "$displacement" "$smoothing_radius" < "$filein" > "$fileout"
}


function apply_data_augmentation_to_file() {
    filename="$1"
    dirin="$2"
    dirout="$3"
    
    filename_base="${filename%%.*}"
    extension="${filename##*.}"

    ppmfile="$TMP_DIR/$filename_base.ppm"
    # The image morph program only accepts ppm files, so any non-ppm files will have to be converted first.
    test "$extension" = "ppm" && cp "$dirin/$filename" "$ppmfile" || convert "$dirin/$filename" "$ppmfile" 

    # The image morph program we use has a maximum value for the smoothing component,
    # As the smoothing value cannot be greater than min(width, height) / 2.5
    ppmfile_dims=$(identify "$ppmfile" | awk '{print $3}')
    width=$(grep -o "^[0-9]*" <<< "$ppmfile_dims")
    height=$(grep -o "[0-9]*$" <<< "$ppmfile_dims")
    min_dim=$(($width > $height ? $height : $width))
    max_smoothing=$(awk "BEGIN {print int($min_dim / 2.5)}")

    smoothing_radius=$(($SETTING_IMAGE_MORPH_SMOOTHING_RADIUS > $max_smoothing ? $max_smoothing : $SETTING_IMAGE_MORPH_SMOOTHING_RADIUS))
 
    for ((i=0; i<$SETTING_IMAGE_MORPH_COUNT; i++)); do
        output_file="$dirout/$filename_base""_$i.ppm"
        ./"$IMAGE_MORPH" "$SETTING_IMAGE_MORPH_DISPLACEMENT" "$smoothing_radius" < "$ppmfile" > "$output_file"
    done

    rm "$ppmfile"
}


function apply_data_augmentation_to_dir() {
    # Remove trailing '/' if it exists.
    dirin="${1%/}"
    # Replace the part in the path (e.g. the '0' in 'dataset/0/Tsadi-final'
    # To the part + the suffix (e.g. to 'dataset/0_augmented/Tsadi-final').
    dirout=$(perl -Xpe "s/(?<=\/[0-9]{1,3})(?=\/)/$AUGMENTED_SUFFIX/" <<< "$dirin")

    mkdir -p "$dirout"

    find "$dirin" -type f -printf "%f\n" | while read -r line; do
        apply_data_augmentation_to_file "$line" "$dirin" "$dirout"
    done
}


export -f ensure_imagemorph_available
export -f run_image_morph
export -f apply_data_augmentation_to_file
export -f apply_data_augmentation_to_dir


mkdir -p "$DATA_AUGMENTATION_DIR"
mkdir -p "$TMP_DIR"


ensure_imagemorph_available

dir_count=$(find "$DATASET_DIR" -type d -links 2 ! -empty | grep -v "$AUGMENTED_SUFFIX" | wc -l)
processed=0

find "$DATASET_DIR" -type d -links 2 ! -empty | grep -v "$AUGMENTED_SUFFIX" | while read -r line; do
    test $(($processed % 5)) -eq 0 && printf "Processed %3d / %3d directories!\n" $processed $dir_count
    apply_data_augmentation_to_dir "$line"
    processed=$(($processed + 1))
done


echo "Done! All directories have been processed!"


test -z $(ls -A "$TMP_DIR") && rm -d "$TMP_DIR"




