from pathlib import Path
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MIN_AGE = 0
MAX_AGE = 30
RNG_DATASET_NAME =str(MIN_AGE)+"-"+str(MAX_AGE)

DATASET_METADATA_PATH = Path.home() / "Projects" / "data" / "anon.xlsx"
TEST_DATA_PATH = Path.home() / "Projects" / "data" / "test_idx.csv"
DATASET_IMAGES_PATH = Path.home() / "Projects" / "data" / "img_jpg"
COMPILED_DATASETS_PATH = Path(__file__).parent.absolute() / "compiled_Datasets"

#Si los conjuntos vienen dados
FIXED_GROUPS = True
FIXED_DATA_PATH = Path.home() / "Projects" / "data" / "fixed_data.csv"


STATIC_TEST = True
STATIC_FOLDS = True

VAL_SPLIT = 0.2

RESIZED_SHAPE = (128, 256)
BATCH_SIZE = 4
N_EPOCHS = 200
SHUFFLE = True
N_WORKERS = 3
OMEGA = 0.0001
MAX_ITER_NO_IMPROVE = 200

N_FOLDS = 8
