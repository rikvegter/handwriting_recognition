# Automatic Handwriting Recognition of Ancient Hebrew in the Dead Sea Scrolls

A Python program that performs OCR on binarized images of the Dead Sea Scrolls,
and determines the style period based on character features.

## Usage

```bash
python main.py --arguments
```

### Arguments

#### General

- `-i , --input_path` _(Required)_

    The path of the image to process

- `-o, --output_path` _(Optional)_

    The path to save output to (default: ./)

- `-d, --debug` _(Optional)_

    Save the results of intermediate steps to a debug directory in the output path (default: False)