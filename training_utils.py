def train(model, loader, age_criterion, sex_criterion, optimizer, device):
    model.train()
    for imgs, age, sex in loader:
        # Mover imgs a device
        # Mover age a device
        # Mover sex a device
        optimizer.zero_grad()
        # Meter as imaxes na rede para sacar a idade e o sexo
        # Calcular o coste en función da idade
        # Calcular o coste en función do sexo
        # Integrar os dous valores en un único valor
        # Executar o backpropagation
        optimizer.step()

    # Devolver o coste da idade e o sexo


def validate(model, loader, age_criterion, sex_criterion):
    model.eval()
    batch_losses_age = 0
    batch_losses_sex = 0
    total_steps = 0  # Número de lotes

    for imgs, age, sex in loader:
        # Mover imgs a device
        # Mover age a device
        # Mover sex a device
        # Meter as imaxes na rede para sacar a idade e o sexo
        # Acumular o coste da idade na variable batch_losses_age
        # Acumular o coste do sexo na variable batch_losses_sex
        total_steps += 1

    # Calcular o coste de idade dividindo a variable batch_losses_age polo numero de lotes
    # Calcular o coste de sex dividindo a variable batch_losses_sex polo numero de lotes

    # Devolver o coste da idade e o sexo
