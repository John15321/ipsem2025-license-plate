# generate_dataset_with_labels

## Description

generate_dataset_with_labels.py is a Python script that automatically processes images containing single characters, recognizes text using the Tesseract OCR library, and organizes them into a structured dataset. The script utilizes the `lpce` tool for license plate extraction. It moves images to a new folder and saves the recognized characters in corresponding text files.

## Requirements

Before running the script, ensure you have the following libraries installed:

- OpenCV (`cv2`)
- Tesseract OCR (`pytesseract`)
- argparse
- shutil
- os

Additionally, Tesseract OCR must be installed and configured on your system. 

## Installation

To install the required packages, run:

```bash
pip install opencv-python pytesseract lpce
```

## Usage

The script can be executed from the command line, optionally specifying input and output folder paths:

```bash
python generate_dataset_with_labels.py --input "path_to_input_folder" --output "path_to_output_folder"
```

If no paths are provided, the default values will be used:
- `test-plates` as the input folder
- `ocr-dataset` as the output folder

### Example Run:

```bash
python generate_dataset_with_labels.py --input "./test-plates" --output "./ocr_dataset"
```

## How the Script Works

1. Uses `lpce` to extract license plate characters from images.
2. Scans the extracted character images for `.png` files.
3. Recognizes text on each image using `pytesseract`.
4. Copies images to the `images` folder within the output directory.
5. Creates `.txt` files in the `labels` folder, saving the recognized characters.
6. Deletes intermediate output folders (`OUTPUT_BOX` and `OUTPUT_SINGLE`) after processing.
7. Notifies the user upon completion and provides paths to the processed data.

