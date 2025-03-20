import sys
from lpce import PlateExtractor, FTYPE, STYPE

# Generating our istance
extractor = PlateExtractor()

# Fetching the user path: can be an image or an entire folder containing images
path = sys.argv[1]

letters = extractor.apply_extraction_onpath(input_path=path, ftype=FTYPE.SINGLECHAR, stype=STYPE.BINARY, ret=True, write=False)

