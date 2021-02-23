from pathlib import Path
import torch
DATASET_METADATA_PATH = Path.home() / "Projects" / "data" / "anon.xlsx"
DATASET_IMAGES_PATH = Path.home() / "Projects" / "data" / "img_jpg"
COMPILED_DATASETS_PATH = Path(__file__).parent.absolute()

RESIZED_SHAPE = (128, 256)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
