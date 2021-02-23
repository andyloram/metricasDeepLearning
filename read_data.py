import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imageio import imread
from tqdm import tqdm

from config import RESIZED_SHAPE, DATASET_METADATA_PATH, DATASET_IMAGES_PATH, COMPILED_DATASETS_PATH
from utils import stratified_cv_split_ttv_age_and_sex
from utils import resize_single_image

home_dir = Path.home()

if not COMPILED_DATASETS_PATH.is_dir():
    COMPILED_DATASETS_PATH.mkdir()

k = 8

age = dict()
bad_age = dict()
bad_quality = dict()
sex = dict()
bad_sex = dict()
anon_dataframe = pd.read_excel(DATASET_METADATA_PATH)
for i, row in anon_dataframe.iterrows():
    idx = int(row["opg_id"])
    age[idx] = row["patient_age"]
    sex[idx] = row["sex"]
    bad_age[idx] = row["bad_age"] in ['x', 'X']
    bad_quality[idx] = row["bad_quality"] in ['x', 'X']
    bad_sex[idx] = sex[idx] == "Unknown"


# Full dataset
def build_full_dataset(name, img_shape, include_bad_quality_images):
    idx_valid = np.array([j for j in age.keys()
                          if not bad_age[j] and not bad_sex[j]
                          and (not bad_quality[j] or include_bad_quality_images)])

    dataset = dict()
    dataset_file_template = "{}_dataset_w_bad_quality.pkl" if include_bad_quality_images else "{}_dataset_wo_bad_quality.pkl"
    ecv_file_template = "{}_ecv_w_bad_quality.pkl" if include_bad_quality_images else "{}_ecv_wo_bad_quality.pkl"
    dataset_file = COMPILED_DATASETS_PATH.joinpath(dataset_file_template.format(name))
    ecv_file = COMPILED_DATASETS_PATH.joinpath(ecv_file_template.format(name))

    if dataset_file.is_file() and ecv_file.is_file():
        print("Both dataset and cross validation indices exist")

        with dataset_file.open("rb") as f:
            dataset = pickle.load(f)
        with ecv_file.open("rb") as f:
            folds = pickle.load(f)

    else:
        print("Calculating folds...")
        folds = stratified_cv_split_ttv_age_and_sex(np.array([age[i] for i in idx_valid]),  # Vector de idades
                                                    np.array([sex[t] for t in idx_valid]),  # Vector de sexos
                                                    k=8,  # Número de particións
                                                    original_idx=idx_valid)  # Identificadores das imaxes

        print("Reading images...")
        for i in tqdm(range(len(idx_valid))):
            idx = idx_valid[i]
            dataset[idx] = dict()
            img_path = DATASET_IMAGES_PATH / f"{idx}.jpg"
            dataset[idx]["img"], dataset[idx]["orig_shape"] = resize_single_image(
                imread(img_path), img_shape)
            dataset[idx]["age"] = age[idx]
            dataset[idx]["sex"] = sex[idx]

        print("{} images collected".format(len(idx_valid)))

        with ecv_file.open("wb") as f:
            pickle.dump(folds, f)

        with dataset_file.open("wb") as f:
            pickle.dump(dataset, f)

    return dataset, folds


def build_dataset_age_range(full_dataset, min_age, max_age, dataset_name, include_bad_quality_images):
    idx_valid = np.array(
        [i for i in full_dataset.keys() if
         not bad_age[i] and min_age <= age[i] < max_age and (not bad_quality[i] or include_bad_quality_images)])
    dataset_file_template = "{}_dataset_w_bad_quality.pkl" if include_bad_quality_images else "{}_dataset_wo_bad_quality.pkl"
    ecv_file_template = "{}_ecv_w_bad_quality.pkl" if include_bad_quality_images else "{}_ecv_wo_bad_quality.pkl"
    dataset_file = COMPILED_DATASETS_PATH.joinpath(dataset_file_template.format(dataset_name))
    ecv_file = COMPILED_DATASETS_PATH.joinpath(ecv_file_template.format(dataset_name))

    print("Building dataset between {:.1f} and {:.1f} years old".format(min_age / 365, max_age / 365))
    if dataset_file.is_file():
        print("Dataset already exists")
        with dataset_file.open("rb") as f:
            dataset = pickle.load(f)
    else:
        print("{} images collected".format(len(idx_valid)))

        dataset = {i: full_dataset[i] for i in idx_valid}

        with dataset_file.open("wb") as f:
            pickle.dump(dataset, f)

    if ecv_file.is_file():
        print("Cross validation indices already exist")
        with ecv_file.open("rb") as f:
            folds = pickle.load(f)
    else:
        print("Building cross validation indices between {:.1f} and {:.1f} years old".format(min_age / 365,
                                                                                             max_age / 365))
        folds = stratified_cv_split_ttv_age_and_sex(age=np.array([age[i] for i in idx_valid]),
                                                    sex=np.array([sex[i] for i in idx_valid]),
                                                    k=8,
                                                    original_idx=idx_valid)
        with ecv_file.open("wb") as f:
            pickle.dump(folds, f)

    return dataset, folds
inc_bad_qual = True

full_dataset_w_bad_quality, full_dataset_w_bad_quality_folds = build_full_dataset(name="all", img_shape=RESIZED_SHAPE,
                                                include_bad_quality_images=inc_bad_qual)
range_dataset_w_bad_quality, range_dataset_w_bad_quality_folds=build_dataset_age_range(full_dataset_w_bad_quality,
                        min_age=15 * 365, max_age=25 * 365,
                        dataset_name="15-25", include_bad_quality_images=inc_bad_qual)
