import torch
from config import DEVICE, OMEGA, BATCH_SIZE

def train(model, loader, age_criterion, sex_criterion, optimizer, writer, fold, steps):
    model.train()
    torch.set_grad_enabled(True)
    total_steps=0
    for i, data in enumerate(loader):
        img, label = data
        age = label[0]
        sex = label[1]


        img = img.to(DEVICE)
        age = age.to(DEVICE)
        sex = sex.to(DEVICE)

        size = len(age)

        optimizer.zero_grad()

        age_out, sex_out = model(img)

        # Calcular o coste en función da idade
        age_out = age_out.reshape([size])
        sex_out = sex_out.reshape([size])

        age_loss = torch.sqrt(age_criterion(age_out, age))
        sex_loss = sex_criterion(sex_out, sex)

        # Integrar os dous valores en un único valor
        comb_loss = (OMEGA * age_loss) + sex_loss
        comb_loss.backward()
        optimizer.step()
        writer.add_scalar('{}-fold Train Total Loss'.format(fold),
                          comb_loss, i+steps)
        writer.add_scalar('{}-fold Train Age Loss'.format(fold),
                          age_loss, i * steps)
        writer.add_scalar('{}-fold Train Sex Loss'.format(fold),
                          sex_loss, i+steps)
        total_steps+=1
        return total_steps


def validate(model, loader, age_criterion, sex_criterion,writer,fold, mode = 'val'):
    model.eval()
    batch_losses_age = 0
    batch_losses_sex = 0
    batch_comb_loss = 0
    total_steps = 0  # Número de lotes
    batch_age_diff_avg = 0
    batch_sex_diff_avg = 0


    age_pred= torch.empty(0).to(DEVICE)
    age_data= torch.empty(0).to(DEVICE)
    sex_pred = torch.empty(0).to(DEVICE)
    sex_data = torch.empty(0).to(DEVICE)

    for i, data in enumerate(loader):
        img, label = data
        age = label[0]
        sex = label[1]
        img = img.to(DEVICE)
        age = age.to(DEVICE)
        sex = sex.to(DEVICE)

        size = len(age)
        # Meter as imaxes na rede para sacar a idade e o sexo
        torch.set_grad_enabled(False)

        age_out, sex_out = model(img)
        age_out = age_out.reshape([size])
        sex_out = sex_out.reshape([size])
        # Acumular o coste da idade na variable batch_losses_age
        batch_losses_age += age_criterion(age_out, age)
        # Acumular o coste do sexo na variable batch_losses_sex
        batch_losses_sex += sex_criterion(sex_out, sex)
        comb_loss = OMEGA * age_criterion(age_out, age) + sex_criterion(sex_out, sex)
        batch_comb_loss += comb_loss

        sex_diff = torch.abs(torch.sub(sex, sex_out))
        age_diff = torch.abs(torch.sub(age, age_out))
        batch_sex_diff_avg += torch.sum(sex_diff).float() / size
        batch_age_diff_avg += torch.sum(age_diff).float() / size
        total_steps += 1
        if mode == 'test':
            age_data = torch.cat((age_data, age), 0)
            age_pred = torch.cat((age_pred, age_out), 0)
            sex_data = torch.cat((sex_data, sex), 0)
            sex_pred = torch.cat((sex_pred, sex_out), 0)
            writer.add_scalar('{}-fold Test Batch Age Diff Mean'.format(fold), torch.sum(torch.sub(age, age_out))/len(age), i)


    # Calcular o coste de idade dividindo a variable batch_losses_age polo numero de lotes
    age_cost = batch_losses_age / total_steps
    # Calcular o coste de sex dividindo a variable batch_losses_sex polo numero de lotes
    sex_cost = batch_losses_sex / total_steps
    total_cost = batch_comb_loss / total_steps

    avg_age_diff = batch_age_diff_avg / total_steps
    avg_sex_diff = batch_sex_diff_avg / total_steps
    # Devolver o coste da idade e o sexo
    return [age_data, age_pred, sex_data, sex_pred, total_cost, age_cost, sex_cost, avg_age_diff, avg_sex_diff]


