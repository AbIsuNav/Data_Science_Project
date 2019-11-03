import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TransitionLayer(nn.Module):
    """
    Insert a transition layer, a global pooling layer, a prediction layer
    (and a loss layer in the end -- written in the loss function WCLE) (after the last convolutional layer in ResNet).
    input: output of last convolutional layer in ResNet (with a size of batch_size x D x S x S)
    output: weighted spatial activation maps for each disease class (with a size of batch_size x S × S × C)
    """
    def __init__(self, input_features, S, D, n_classes, r=0):
        super(TransitionLayer, self).__init__()
        self.S = S
        self.r = r
        # After conv1 dim=(1, D, S, S)
        self.conv1 = nn.Conv2d(input_features, D, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(D)
        # After fc dim=(1, n_classes)
        self.fc = nn.Linear(D, n_classes)

    def forward(self, x, CAM=False):
        x = self.conv1(x)
        x = self.bn(x)
        if not CAM:  # classification
            # After global pool dim=
            x = self.global_pool(x, app=True)
            out = torch.exp(self.fc(x))
            out = out / out.sum(-1).view(-1, 1)
        else: # class activation map -- for heatmap
            out = [torch.einsum(("ab, bcd ->acd"), (self.fc.weight.data, x[i])) for i in range(x.size(0))]

        return out

    def global_pool(self, x, app=False):
        # Log-Sum-Exp(LSE) pooling layer
        if app:
        # Eq(3) version, used in Xray paper
            x_star = torch.abs(x).max(dim=-1)[0].max(dim=-1)[0]
            x_p = x_star + 1/self.r*torch.log(1/self.S*torch.exp(self.r*(x-x_star.unsqueeze(-1).unsqueeze(-1))).sum(-1).sum(-1))
        else:
        # Eq(2)
            x_p = 1/self.r*torch.log(1/self.S*torch.exp(x*self.r).sum(-1).sum(-1))

        return x_p



if __name__=='__main__':
    # shape of the output of last conv layer in ResNet
    S = 8
    D = 2048
    n_classes = 8
    input_features = D
    r = 10  # hyperparameter for global pooling layer in the transition layer/model
    batch_size = 3
	# test
    sample_input = torch.ones(size=(batch_size, D, S, S), requires_grad=False)
    transition = TransitionLayer(input_features, S, D, n_classes, r=r)
    out = transition(sample_input, CAM=False)
    for i in range(len(out)):
        for c in range(n_classes):
            plt.pcolormesh(out[i][c].detach().numpy())
            plt.title('class activation map{}{}'.format(i, c))
            plt.show()








