from pathlib import Path

DATASET_METADATA_PATH = Path.home() / "Nextcloud" / "OPG" / "anon.xlsx"
DATASET_IMAGES_PATH = Path.home() / "Nextcloud" / "OPG" / "img_jpg"
COMPILED_DATASETS_PATH = Path(__file__).parent.absolute()

RESIZED_SHAPE = (128, 256)
DEVICE = "cuda"
