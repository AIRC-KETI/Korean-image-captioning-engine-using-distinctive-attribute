import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(1)

class AttMLP(nn.Module):
    def __init__(self, resnet_out_size, hidden_size, vocab_size):
        super(AttMLP, self).__init__()

        resnet = models.resnet152(pretrained=True)
        resnet_module = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*resnet_module)

        self.layer1 = nn.Sequential(
            nn.Linear(resnet_out_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size, momentum=0.01),
            nn.ReLU())

    def forward(self, images):

        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        out = self.layer1(features)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class AttMLP_saved(nn.Module):
    def __init__(self, resnet_out_size, hidden_size, vocab_size):
        super(AttMLP_saved, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(resnet_out_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size, momentum=0.01),
            nn.ReLU())

    def forward(self, features):
        features = features.view(features.size(0), -1)
        out = self.layer1(features)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out