import torch
import dataset_utils as dat_ut
from dataset_utils import N_FOLDS
import model as m
from config import DEVICE, N_EPOCHS, RNG_DATASET_NAME, MAX_ITER_NO_IMPROVE, STATIC_TEST
import training_utils as train_ut
import torch.nn as nn
from utils import print_metrics
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

file_name = RNG_DATASET_NAME
RESULTS_WEIGHTS_PATH = Path(__file__).parent.absolute() / file_name
if not RESULTS_WEIGHTS_PATH.is_dir():
    RESULTS_WEIGHTS_PATH.mkdir()



def main():
    total_steps=0
    results_file = open(RESULTS_WEIGHTS_PATH.joinpath("results.txt"), "w")

    age_criterion = nn.MSELoss()
    # sex_criterion = nn.NLLLoss()
    sex_criterion = nn.BCELoss();

    age_pred = torch.empty(0).to(DEVICE)
    age_data = torch.empty(0).to(DEVICE)
    sex_pred = torch.empty(0).to(DEVICE)
    sex_data = torch.empty(0).to(DEVICE)

    for i in range(N_FOLDS):
        model = m.Dasnet().to(DEVICE)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06)

        if STATIC_TEST:
            training_gen, eval_gen, test_gen = dat_ut.kfold_generator_simple(i)
        else:
            training_gen, eval_gen, test_gen = dat_ut.kfold_generator_simple(i)



        best_epoch_result = [-1, -1, -1, -1]
        best_epoch = 0

        best_epoch_model = dict()
        no_upgrade_cont = 0


        print("\n{}-FOLD of {}".format(i + 1, N_FOLDS), file=results_file)
        print("{}-FOLD of {}".format(i + 1, N_FOLDS))

        for k in range(1, N_EPOCHS + 1):

            steps = train_ut.train(model, training_gen, age_criterion, sex_criterion, optimizer, writer, i, total_steps)
            total_steps+=steps
            _, _, _, _, total_loss, age_loss, sex_loss, avg_age_diff, avg_sex_diff = train_ut.validate(model, eval_gen,
                                                                                                       age_criterion,
                                                                                                       sex_criterion, writer, i)
            writer.add_scalar('{}-fold Validation Total Loss'.format(i), total_loss, k)
            writer.add_scalar('{}-fold Validation Age Loss'.format(i), age_loss, k)
            writer.add_scalar('{}-fold Validation Sex Loss'.format(i), sex_loss, k)
            writer.add_scalar('{}-fold Validation Average Age Diff'.format(i), avg_age_diff, k)
            writer.add_scalar('{}-fold Validation Average Sex Diff'.format(i), avg_sex_diff, k)

            #print(
                #"{}/{} EPOCH : TOTAL LOSS = {} / AGE_LOSS = {} / SEX_LOSS = {} / AVG_AGE_DIFF = {} / AVG_SEX_DIFF = {}".format(
                 #   k, N_EPOCHS, total_loss, age_loss, sex_loss, avg_age_diff, avg_sex_diff), file=log_file)

            if best_epoch_result[0] >= total_loss or best_epoch_result[0] == -1:
                best_epoch_result = [total_loss, age_loss, sex_loss, avg_age_diff, avg_sex_diff]
                best_epoch_model = model.state_dict()
                best_epoch = k
                no_upgrade_cont = 0

            if best_epoch_result[0] < total_loss:
                no_upgrade_cont += 1

            if no_upgrade_cont == MAX_ITER_NO_IMPROVE:
                print("UPGRADE FIN / EPOCH: {}".format(best_epoch), file=results_file)
                break

        model.load_state_dict(best_epoch_model)
        #print(
            #"BEST EPOCH RESULT :: {}/{} EPOCH: TOTAL LOSS = {} / AGE_LOSS = {} / SEX_LOSS = {} / AVG_AGE_DIFF = {} / AVG_SEX_DIFF = {}".format(
            #best_epoch, N_EPOCHS, best_epoch_result[0], best_epoch_result[1], best_epoch_result[2],
            #best_epoch_result[3], best_epoch_result[4]), file=log_file)

        torch.save(model.state_dict(), RESULTS_WEIGHTS_PATH.joinpath("{}_fold_model.pth".format(i)))
        age, age_out, sex, sex_out, total_test_loss, age_test_loss, sex_test_loss, avg_age_diff, avg_sex_diff = train_ut.validate(
            model, test_gen, age_criterion, sex_criterion,writer, i, 'test')

        print("TEST :: TOTAL LOSS = {} / AGE_LOSS = {} / SEX_LOSS = {} / AVG_AGE_DIFF = {} / AVG_SEX_DIFF = {}".format(
            total_test_loss, age_test_loss, sex_test_loss, avg_age_diff, avg_sex_diff), file=results_file)

        age_data = torch.cat((age_data, age), 0)
        age_pred = torch.cat((age_pred, age_out), 0)
        sex_data = torch.cat((sex_data, sex), 0)
        sex_pred = torch.cat((sex_pred, sex_out), 0)

        me = print_metrics(age, age_out, sex, sex_out, results_file)
        writer.add_scalar('{}-fold Test Error Age Diff'.format(i), me, i)

    print_metrics(age_data, age_pred, sex_data, sex_pred, results_file)

    results_file.close()
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
