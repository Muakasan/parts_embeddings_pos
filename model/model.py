#Aidan San
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, part, embed_dim=300, hidden_dim=256, num_splits=5, num_tags=12):
        super(Net, self).__init__()
        input_dim = -1
        if part:
            input_dim = int(embed_dim/num_splits)
        else:
            input_dim = embed_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
