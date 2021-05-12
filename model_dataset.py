import torch
from torch.utils.data import Dataset
import numpy as np
from config import RESIZED_SHAPE


class ModelDataset(Dataset):

    def __init__(self, raw_data, fold_idx, data_keys):
        #Inicializamos los arrays de las imágenes el sexo y la edad
        load_img = np.zeros((len(fold_idx), 1, int(RESIZED_SHAPE[0]), int(RESIZED_SHAPE[1])))
        load_age = np.zeros((len(fold_idx)))
        load_sex = np.zeros((len(fold_idx)))

        #Introduciomes los datos de forma que los fold_idx nos da la posición en el array de claves de data_keys
        #Debemos de entrar en el valor de data_keys para cargar los atributos
        for i in range(len(fold_idx)):
            index = int(data_keys[int(fold_idx[i])])

            load_img[i][0] = raw_data[index]['img']

            load_age[i] = raw_data[index]['age']

            if raw_data[index]['sex'] == "V":
                load_sex[i] = 1.0
            else:
                load_sex[i] = 0.0
        #Transformamos a tensores y los convertimos a tipo float ¿Necesario?, los captaba como doucles y daba error
        self.img = torch.from_numpy(load_img).float()
        self.age = torch.from_numpy(load_age).float()
        self.sex = torch.from_numpy(load_sex).float()


    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = (self.age[idx], self.sex[idx])
        return img, label

