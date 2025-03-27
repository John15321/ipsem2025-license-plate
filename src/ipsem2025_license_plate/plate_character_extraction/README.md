# License Plate Character Extraction

This package provides tools for processing license plate images, correcting their perspective, extracting characters, and generating labeled datasets for OCR training.

## Scripts

### 1. Correct Perspective (`correct_perspective.py`)

Corrects the perspective of license plate images using various methods.

#### Usage

```bash
python -m ipsem2025_license_plate.plate_character_extraction.correct_perspective --input-folder INPUT_FOLDER --output-folder OUTPUT_FOLDER [--method {white,superpixel,default}] [--recursive] [--verbose]
```

#### Parameters

- `--input-folder`: Path to the folder containing license plate images
- `--output-folder`: Path where the corrected images will be saved
- `--method`: Method used for perspective correction
  - `white` (default): Detects white regions to determine plate boundaries
  - `superpixel`: Uses a superpixel-based approach for segmentation
  - `default`: Uses a generic unwarping method based on contours
- `--recursive`: Process all images in subdirectories recursively
- `--verbose`: Enable detailed logging output

### 2. Extract Images from Plates (`extract_images_form_plates.py`)

Extracts individual characters from corrected license plate images.

#### Usage

```bash
python -m ipsem2025_license_plate.plate_character_extraction.extract_images_form_plates --input-folder INPUT_FOLDER --output-folder OUTPUT_FOLDER [--min-area MIN_AREA] [--max-area MAX_AREA] [--recursive] [--verbose]
```

#### Parameters

- `--input-folder`: Path to the folder containing corrected license plate images
- `--output-folder`: Path where the extracted character images will be saved
- `--min-area`: Minimum area for character contours (filters out noise)
- `--max-area`: Maximum area for character contours (filters out large regions)
- `--recursive`: Process all images in subdirectories recursively
- `--verbose`: Enable detailed logging output

### 3. Generate Dataset with Labels (`generate_dataset_with_labels.py`)

Creates a labeled dataset from license plate images by automatically applying perspective correction, extracting characters, and performing OCR recognition.

#### Usage

```bash
ipsem2025-character-extractor --input INPUT_FOLDER --output OUTPUT_FOLDER
```

#### Parameters

- `--input`: Path to the folder containing license plate images (default: 'test-plates')
- `--output`: Path where the labeled dataset will be saved (default: 'ocr-dataset')

#### Output Structure

The output is organized by original license plate names. For each license plate, the following structure is created:

```
output_folder/
  license_plate_1/
    images/
      1.png  # First character from the plate
      2.png  # Second character from the plate
      ...
    labels/
      1.txt  # OCR text for first character
      2.txt  # OCR text for second character
      ...
  license_plate_2/
    images/
      ...
    labels/
      ...
  ...
```

#### Process

1. Automatically applies perspective correction to each license plate
2. Extracts individual characters from the corrected plates
3. Performs OCR recognition on each character
4. Organizes the results by original license plate name

## Workflow Example

1. Correct the perspective of license plate images:
   ```bash
   python -m ipsem2025_license_plate.plate_character_extraction.correct_perspective --input-folder ./raw_plates --output-folder ./corrected_plates --method white --recursive
   ```

2. Extract characters from the corrected plates:
   ```bash
   python -m ipsem2025_license_plate.plate_character_extraction.extract_images_form_plates --input-folder ./corrected_plates --output-folder ./extracted_characters --min-area 200 --max-area 5000
   ```

3. Generate a labeled dataset for OCR training:
   ```bash
   ipsem2025-character-extractor --input ./raw_plates --output ./labeled_dataset
   ```

   Or alternatively, use the individual script:
   ```bash
   python -m ipsem2025_license_plate.plate_character_extraction.generate_dataset_with_labels --input ./raw_plates --output ./labeled_dataset
   ```

## Notes

- The scripts will create output directories if they don't exist
- Failed operations are logged to the console
- For optimal results, use high-quality license plate images
- The `generate_dataset_with_labels.py` script now includes perspective correction automatically
- Character extraction from license plates is now organized by the original license plate names