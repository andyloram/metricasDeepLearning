import torch
import model as m
import config as conf
import numpy as np
import read_data as rdata
import model_dataset as mdataset
import training_utils as train_ut
import torch.nn as nn
# dataset = rdata.full_dataset_w_bad_quality
# dataset_folds = rdata.full_dataset_w_bad_quality_folds
dataset = rdata.range_dataset_w_bad_quality
dataset_folds = rdata.range_dataset_w_bad_quality_folds

def main():
    # Params
    n_epochs = 100
    validation_split = 0.2
    shuffle_dataset = True
    age_criterion = nn.MSELoss()
    # sex_criterion = nn.NLLLoss()
    sex_criterion = nn.BCELoss();
    random_seed = 42
    model = m.Dasnet()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06)

    data_keys = np.array(list(dataset.keys()), dtype=np.dtype)

    # Partition [ train+eval , test]
    partition = dict()
    train_data_idx = np.array([])

    for i in range(len(dataset_folds)):
        for j in range(len(dataset_folds)):
            if j != i:
                train_data_idx = np.append(train_data_idx, [int(dataset_folds[j][k]) for k in range(len(dataset_folds[j]))])

        partition['tr_ev'] = train_data_idx
        partition['test'] = np.array(dataset_folds[i])

        data_test_eval = mdataset.ModelDataset(dataset, partition['tr_ev'], data_keys)
        test_dataset = mdataset.ModelDataset(dataset, partition['test'], data_keys)

        train_sampler, eval_sampler = train_ut.train_eval_split(data_test_eval, random_seed, validation_split, shuffle_dataset)

        #Generación de los dataloaders
        training_gen = torch.utils.data.DataLoader(data_test_eval, batch_size=conf.batch_size, sampler=train_sampler,
                                                   shuffle=False, num_workers=1)
        eval_gen = torch.utils.data.DataLoader(data_test_eval, batch_size=conf.batch_size, sampler=eval_sampler,
                                               shuffle=False, num_workers=1)
        for k in range(1, n_epochs+1):
            train_loss = train_ut.train(model, training_gen, age_criterion, sex_criterion, optimizer, conf.DEVICE)
            total_cost, age_cost, sex_cost = train_ut.validate(model, eval_gen, age_criterion, sex_criterion, conf.DEVICE)
            print("TOTAL COST: {}", total_cost)









if __name__ == "__main__":
    main()
