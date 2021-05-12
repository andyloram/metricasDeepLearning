import pickle
from pathlib import Path
import math
import numpy as np
import pandas as pd
from imageio import imread
from tqdm import tqdm

from config import RESIZED_SHAPE, DATASET_METADATA_PATH, DATASET_IMAGES_PATH, COMPILED_DATASETS_PATH, \
    MAX_AGE, MIN_AGE, RNG_DATASET_NAME, STATIC_TEST, TEST_DATA_PATH, N_FOLDS, STATIC_FOLDS, FIXED_GROUPS, FIXED_DATA_PATH
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

if FIXED_GROUPS:
    test_idx = []
    eval_idx = []
    train_idx =[]
    fixed_dataframe = pd.read_csv(FIXED_DATA_PATH)
    test_idx = np.array(fixed_dataframe.iloc[:, 1].to_list())
    eval_idx = np.array(fixed_dataframe.iloc[:, 2].to_list())
    train_idx = np.array(fixed_dataframe.iloc[:, 3].to_list())
    test_idx = [x for x in test_idx if math.isnan(x) == False]
    eval_idx = [x for x in eval_idx if math.isnan(x) == False]
    train_idx = [x for x in train_idx if math.isnan(x) == False]

elif STATIC_TEST:
    test_idx =[]
    test_dataframe = pd.read_csv(TEST_DATA_PATH)
    test_idx = np.array(test_dataframe.iloc[:, 1].to_list())
    test_idx = [x for x in test_idx if math.isnan(x) == False]
    fold_values = {k: [] for k in list(range(N_FOLDS))}
    if STATIC_FOLDS:
        for k in range(len(fold_values)):
            aux_fold=np.array(test_dataframe.iloc[:, k+2].to_list())
            fold_values[k] = [x for x in aux_fold if math.isnan(x) == False]



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
                                                    k=N_FOLDS,  # Número de particións
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



        with ecv_file.open("wb") as f:
            pickle.dump(folds, f)

        with dataset_file.open("wb") as f:
            pickle.dump(dataset, f)

    print("{} images collected".format(len(idx_valid)))
    return dataset, folds


def build_dataset_age_range(full_dataset, min_age, max_age, dataset_name, include_bad_quality_images):
    idx_valid = np.array(
        [i for i in full_dataset.keys() if
         not bad_age[i] and min_age <= age[i] < max_age and (not bad_quality[i] or include_bad_quality_images)])
    idx_folds = np.copy(idx_valid)

    if STATIC_TEST==True:
        remove_idx=[]
        for i in range(len(idx_valid)):
            for j in test_idx:
                if idx_valid[i] == j:
                    remove_idx.append(i)
        idx_folds =np.delete(idx_valid, remove_idx)


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

        dataset = {i: full_dataset[i] for i in idx_valid}

        with dataset_file.open("wb") as f:
            pickle.dump(dataset, f)

    print("{} images collected".format(len(idx_valid)))

    if STATIC_TEST:
        print("{} images for train-eval".format(len(idx_folds)))
        print("{} images for test".format(len(idx_valid)-len(idx_folds)))
    if ecv_file.is_file():
        print("Cross validation indices already exist")
        with ecv_file.open("rb") as f:
            folds = pickle.load(f)
    else:
        print("Building cross validation indices between {:.1f} and {:.1f} years old".format(min_age / 365,
                                                                            max_age / 365))
        folds = {k: [] for k in list(range(N_FOLDS))}

        if STATIC_FOLDS == False:
            folds_aux = stratified_cv_split_ttv_age_and_sex(age=np.array([age[i] for i in idx_folds]),
                                                        sex=np.array([sex[i] for i in idx_folds]),
                                                        k=N_FOLDS,
                                                        original_idx=idx_folds)

            aux_dataset = {i: full_dataset[i] for i in idx_folds}
            for k in range(len(folds_aux)):
                for t in folds_aux[k]:
                    fold_values[k].append(list(aux_dataset.keys())[t])

        for k in range(len(fold_values)):
            for i in range(len(dataset.keys())):
                    for t in range(len(fold_values[k])):
                        if fold_values[k][t]==list(dataset.keys())[i]:
                            folds[k].append(i)



        with ecv_file.open("wb") as f:
            pickle.dump(folds, f)

    return dataset, folds


inc_bad_qual = True

full_dataset_w_bad_quality, full_dataset_w_bad_quality_folds = build_full_dataset(name="all", img_shape=RESIZED_SHAPE,
                                                                                 include_bad_quality_images=inc_bad_qual)
if FIXED_GROUPS == False:
    range_dataset_w_bad_quality, range_dataset_w_bad_quality_folds = build_dataset_age_range(full_dataset_w_bad_quality,
                                                                                         min_age=MIN_AGE * 365,
                                                                                         max_age=MAX_AGE * 365,
                                                                                         dataset_name=RNG_DATASET_NAME,
                                                                                         include_bad_quality_images=inc_bad_qual)
