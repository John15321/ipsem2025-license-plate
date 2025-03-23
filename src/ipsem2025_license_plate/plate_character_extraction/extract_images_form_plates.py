import sys

from tools import FTYPE, STYPE, PlateExtractor

# Generating our istance
extractor = PlateExtractor()

# Fetching the user path: can be an image or an entire folder containing images
path = sys.argv[1]

letters = extractor.apply_extraction_onpath(
    input_path=path, ftype=FTYPE.SINGLECHAR, stype=STYPE.BINARY, ret=True, write=False
)
