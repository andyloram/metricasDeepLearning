import torch
import model as m
from config import DEVICE

def main():
    x = torch.ones(1, 1, 128, 256).to(DEVICE)
    net = m.Dasnet().to(DEVICE)
    [y, t] = net.forward(x)
    print(y.size())
    print(t.size())


if __name__ == "__main__":
    main()


