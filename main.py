import torch
import model as m


def main():
    x = torch.ones(1, 1, 128, 256)
    net = m.Dasnet()
    [y, t] = net.forward(x)
    print(y.size())
    print(t.size())


if __name__ == "__main__":
    main()


