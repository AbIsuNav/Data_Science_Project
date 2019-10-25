import torch
import torch.nn as nn
import matplotlib
from matplotlib import pyplot as plt

class transitionLayer(nn.Module):
	"""
	Insert a transition layer, a global pooling layer, a prediction layer
	and a loss layer in the end (after the last convolutional layer in ResNet). 
	"""

    def __init__(self, input_features, S, D, n_classes, CAM, r=0):
        super(transitionLayer, self).__init__()
        self.S = S
        self.CAM = CAM # class activation maps
        self.r = r

        # After conv1 dim=(1, D, S, S)
        self.conv1 = nn.Conv2d(input_features, D, kernel_size=3, stride=1, padding=1, bias=False)
        # After fc dim=(1, n_classes)
        self.fc = nn.Linear(D, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        if not self.CAM:
            # After global pool dim=
            x = self.global_pool(x)
            out = torch.exp(self.fc(x))
            out = out/out.sum(-1).view(-1, 1)
        else:
            out = [torch.einsum(("ab, bcd ->acd"), (self.fc.weight.data, x[i])) for i in range(x.size(0))]

        return out

    def global_pool(self, x):
        return 1/self.r*torch.log(1/self.S*torch.exp(x*self.r).sum(-1).sum(-1))

if __name__=='__main__':
    S = 8
    D = 2048
    n_classes = 8
    input_features = D
    r = 10
	# test
    sample_input = torch.ones(size=(10, D, S, S), requires_grad=False)
    transition = transitionLayer(input_features, S, D, n_classes, CAM=True, r=r)
    out = transition(sample_input)
    for c in range(n_classes):
        plt.pcolormesh(out[c].detach().numpy())
        plt.show()








