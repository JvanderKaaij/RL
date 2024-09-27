import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

# https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html

def main():
    a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
    print(a)
    b = torch.sin(a)
    # plt.plot(a.detach(), b.detach())
    c = 2 * b
    print(c)
    d = c + 1
    print(d)
    out = d.sum()
    print(out)
    out.backward()
    print(a.grad)
    plt.plot(a.detach(), a.grad.detach())
    plt.show()



if __name__ == "__main__":
    main()