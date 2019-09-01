import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

USE_CUDA = True

# <PAD>: 0
# <BOS>: 1
# <EOS>: 2
# <UNK>: 3
PAD = 0
BOS = 1
EOS = 2
UNK = 3

class InstanceNorm(nn.Module):
    def __init__(self, latent_size):
        super(InstanceNorm, self).__init__()
        self.gamma = nn.Linear(latent_size, 1)
        self.beta = nn.Linear(latent_size, 1)
    def forward(self, x):
        eps = 1e-5
        mean = x.mean((2,3))
        std = x.std((2,3))
        x_norm = (x - mean)/(std + eps)
        gamma = self.gamma(x)
        beta = self.beta(x)
        return x_norm * gamma + beta

class TextEncoder(nn.Module):
    def __init__(self, latent_size, embedding_size, rnn_size):
        super(TextEncoder, self).__init__()
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        def my_conv(kernel_size, rnn_size, embedding_size, latent_size):
            return nn.Sequential(
                nn.Conv2d(1, 100, (kernel_size, embedding_size), padding=(kernel_size-1, 0)),
                InstanceNorm(latent_size)
                nn.Tanh()
                nn.MaxPool2d((rnn_size,1))
                # shape = (N, 100, 1, 1)
            )
        self.conv1 = my_conv(3, self.rnn_size, self.embedding_size, self.latent_size)
        self.conv2 = my_conv(4, self.rnn_size, self.embedding_size, self.latent_size)
        self.conv3 = my_conv(5, self.rnn_size, self.embedding_size, self.latent_size)
    def forward(self, input):
        # input.shape = (N, rnn_size, embedding_size, 1)
        x1 = self.conv1(input) # shape = (N, 1, 1, 100)
        x2 = self.conv2(input)
        x3 = self.conv3(input)
        x1 = x1.view(-1, 100)
        x2 = x2.view(-1, 100)
        x3 = x3.view(-1, 100)
        output = torch.cat((x1, x2, x3), 1)
        return output # shape = (N, 300)

class TextDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, dropout_rate=0):
        super(TextDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.lstm = nn.LSTM(embedding_size, hidden_size, dropout=self.dropout_rate)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        # input.shape = (N, rnn_size=1)
        embedded = self.embedding(input.t()) # shape = (rnn_size=1, N, embedding_size)
        output, hidden = self.lstm(embedded, hidden)
        output = F.log_softmax(self.out(output.squeeze(0)), dim=1)
        return output, hidden




class Generator(nn.Module):
    def __init__(self, hidden_size, embedding_size, rnn_size, latent_size, embedding):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.latent_size = latent_size
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(embedding)
        def my_conv(kernel_size, rnn_size, embedding_size, latent_size):
            return nn.Sequential(
                nn.Conv2d(1, 100, (kernel_size,embedding_size), padding=(kernel_size-1,0)),
                InstanceNorm(latent_size)
                nn.Tanh()
                nn.MaxPool2d((rnn_size,1))
                # shape = (N, 100, 1, 1)
            )
        self.conv1 = my_conv(3, self.rnn_size, self.embedding_size, self.latent_size)
        self.conv2 = my_conv(4, self.rnn_size, self.embedding_size, self.latent_size)
        self.conv3 = my_conv(5, self.rnn_size, self.embedding_size, self.latent_size)
        
    def forward_1(self, input_x, latent_z):
        embedded_x = self.embedding(input_x).view(-1, rnn_size, embedding_size, 1)
        x1 = self.conv1(input_x) # shape = (N, 1, 1, 100)
        x2 = self.conv2(input_x)
        x3 = self.conv3(input_x)
        x1 = x1.view(-1, 100)
        x2 = x2.view(-1, 100)
        x3 = x3.view(-1, 100)
        output = torch.cat((x1, x2, x3), 1)
        return output # shape = (N, 300)

    def forward_2(self, input_x, latent_z, feature_vec):
        embedded_x = self.embedding(input_x.t()) # shape = (rnn_size, N, embedding_size)
        h0 = feature_vec.view(1, -1, self.hidden_size) # shape = (1, N, hidden_size)
        c0 = feature_vec.view(1, -1, self.hidden_size) # shape = (1, N, hidden_size)
        output, (hn, cn) = self.lstm( embedded_x, (h0, c0) )
        return output


