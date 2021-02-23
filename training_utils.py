import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
omega = 10 ** (-3)
def train(model, loader, age_criterion, sex_criterion, optimizer, device):
    model.train()
    torch.set_grad_enabled(True)
    total_steps = 0
    batch_train_loss = 0
    for i, data in enumerate(loader):
        img, label = data
        age = label[0]
        sex = label[1]

        img = img.to(device)
        age = age.to(device)
        sex = sex.to(device)

        optimizer.zero_grad()

        age_out, sex_out = model(img)

        # Calcular o coste en función da idade
        age_out = torch.squeeze(age_out)
        age_loss = torch.sqrt(age_criterion(age_out, age))

        # Calcular o coste en función do sexo
        sex_out = torch.squeeze(sex_out)
        sex_loss = sex_criterion(sex_out, sex)
        # Integrar os dous valores en un único valor
        comb_loss = omega * age_loss + sex_loss
        comb_loss.backward()
        optimizer.step()
        batch_train_loss += comb_loss
        total_steps += 1

    avg_batch_loss = batch_train_loss/total_steps
    return avg_batch_loss


def validate(model, loader, age_criterion, sex_criterion, device):
    model.eval()
    batch_losses_age = 0
    batch_losses_sex = 0
    batch_comb_loss = 0
    total_steps = 0  # Número de lotes

    for i, data in enumerate(loader):
        img, label = data
        age = label[0]
        sex = label[1]
        img = img.to(device)
        age = age.to(device)
        sex = sex.to(device)

        # Meter as imaxes na rede para sacar a idade e o sexo
        torch.set_grad_enabled(False)
        age_out, sex_out = model(img)
        # Acumular o coste da idade na variable batch_losses_age
        batch_losses_age += age_criterion(age_out, age)
        # Acumular o coste do sexo na variable batch_losses_sex
        batch_losses_sex += sex_criterion(sex_out, sex)
        batch_comb_loss += omega * age_criterion(age_out, age) + sex_criterion(sex_out, sex)
        total_steps += 1

    # Calcular o coste de idade dividindo a variable batch_losses_age polo numero de lotes
    age_cost = batch_losses_age / total_steps
    # Calcular o coste de sex dividindo a variable batch_losses_sex polo numero de lotes
    sex_cost = batch_losses_sex / total_steps
    total_cost = batch_comb_loss / total_steps

    # Devolver o coste da idade e o sexo
    return [total_cost, age_cost, sex_cost]

def train_eval_split(data_test_eval, random_seed, validation_split, shuffle_dataset):
    data_test_eval_size = len(data_test_eval)
    print(data_test_eval_size)
    indices = list(range(data_test_eval_size))
    split = int(np.floor(validation_split * data_test_eval_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, eval_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    eval_sampler = SubsetRandomSampler(eval_indices)
    return [train_sampler,eval_sampler]