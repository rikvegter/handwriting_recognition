# Automatic Handwriting Recognition of Ancient Hebrew in the Dead Sea Scrolls

A Python program that performs OCR on binarized images of the Dead Sea Scrolls,
and determines the style period based on character features.

## Usage

Start by cloning the directory:

```bash
git clone https://github.com/rikvegter/handwriting_recognition.git
```

### Using Docker

Running the tool using [Docker](https://www.docker.com/) ensures it will work as
intended. Before trying this, make sure Docker is
[installed](https://www.docker.com/get-started) and and running.

You can run the tool using the following command:

```bash
./run.sh /path/to/input
```

When the script is run for the first time, it will download a complete Docker
image (~2GB), this might take some time. When the image is downloaded, it will
be executed on the images in the folder. The output will be saved to `./results`.

### Running locally

You can also run the python code directly. If you want to do this, you need to
provide the correct command line arguments:

```bash
python main.py 
    [-h]                            # Show help for CLI arguments
    -i str                          # Directory containing files to process. If --single is set, the path to the file.
    [-o str]                        # Output directory
    [--stop_after {1,2,3,4,5}]      # Stop after a given step
    [-d]                            # Save intermediate images to a debug folder in the output directory
    [-s]                            # Process only a single image
    [--classifier str]              # The directory containing the trained character recognition model 
    [--ngram_file str]              # The file containing the .xlsx file with n-gram information 
```

Options within square brackets are optional.

An example of a simple valid input is the following:

```bash
python main.py -i /path/to/images
```

A more complex example would be the following:

```bash
python main.py --single -i /path/to/images/image-2.jpg -o ./custom/results/folder/ -d --stop-after 3
```

This will process only one image, save it to a custom output location, save
debugging images and stop after step 3 (character recognition)
