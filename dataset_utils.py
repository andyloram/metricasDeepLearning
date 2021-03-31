import torch
import numpy as np
import model_dataset as mdataset
import read_data as rdata
from config import VAL_SPLIT, SHUFFLE, BATCH_SIZE, N_WORKERS, STATIC_TEST
from torch.utils.data.sampler import SubsetRandomSampler

dataset = rdata.range_dataset_w_bad_quality
dataset_folds = rdata.range_dataset_w_bad_quality_folds
data_keys = np.array(list(dataset.keys()), dtype=np.dtype)
random_seed = 42
N_FOLDS = len(dataset_folds)
if STATIC_TEST:
    from read_data import test_idx
    test_data_idx_order = np.array([])
    for i in range(len(data_keys)):
        for j in range(len(test_idx)):
            if data_keys[i]==test_idx[j]:
                test_data_idx_order=np.append(test_data_idx_order,i)



def kfold_generator(index):
    partition = dict()
    train_data_idx = np.array([])
    if STATIC_TEST:
        for j in range(N_FOLDS):
                train_data_idx = np.append(train_data_idx,
                                           [int(dataset_folds[j][k]) for k in range(len(dataset_folds[j]))])
        partition['test'] = test_data_idx_order
    else:
        for j in range(N_FOLDS):
            if j != index:
                train_data_idx = np.append(train_data_idx, [int(dataset_folds[j][k]) for k in range(len(dataset_folds[j]))])
        partition['test'] = np.array(dataset_folds[index])

    partition['tr_ev'] = train_data_idx
    data_train_eval = mdataset.ModelDataset(dataset, partition['tr_ev'], data_keys)
    test_dataset = mdataset.ModelDataset(dataset, partition['test'], data_keys)

    train_sampler, eval_sampler = train_eval_split(data_train_eval)

    # Generación de los dataloaders
    training_gen = torch.utils.data.DataLoader(data_train_eval, batch_size=BATCH_SIZE, sampler=train_sampler,
                                               shuffle=False, num_workers=N_WORKERS)
    eval_gen = torch.utils.data.DataLoader(data_train_eval, batch_size=BATCH_SIZE, sampler=eval_sampler,
                                           shuffle=False, num_workers=N_WORKERS)
    test_gen = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=N_WORKERS)

    return training_gen, eval_gen, test_gen


def kfold_generator_simple(index):
    partition = dict()
    train_data_idx = np.array([])
    if STATIC_TEST == False:
        for j in range(len(dataset_folds)):
            t = index + 1
            if index == len(dataset_folds) - 1:
                t = 0
            if j != index and j != t:
                train_data_idx = np.append(train_data_idx,
                                           [int(dataset_folds[j][k]) for k in range(len(dataset_folds[j]))])
        partition['test'] = np.array(dataset_folds[t])
    else:
        for j in range(len(dataset_folds)):
            if j != index:
                train_data_idx = np.append(train_data_idx,
                                           [int(dataset_folds[j][k]) for k in range(len(dataset_folds[j]))])
        partition['test'] = test_data_idx_order

    partition['train'] = train_data_idx
    partition['eval'] = np.array(dataset_folds[index])

    train_dataset = mdataset.ModelDataset(dataset, partition['train'], data_keys)
    test_dataset = mdataset.ModelDataset(dataset, partition['test'], data_keys)
    eval_dataset = mdataset.ModelDataset(dataset, partition['eval'], data_keys)

    training_gen = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=N_WORKERS)
    eval_gen = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    test_gen = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=N_WORKERS)

    return training_gen, eval_gen, test_gen


def train_eval_split(data_test_eval):
    data_test_eval_size = len(data_test_eval)
    indices = list(range(data_test_eval_size))
    split = int(np.floor(VAL_SPLIT * data_test_eval_size))
    if SHUFFLE:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, eval_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    eval_sampler = SubsetRandomSampler(eval_indices)
    return [train_sampler, eval_sampler]
