import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        latent_vector = h_n[-1]
        return latent_vector


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        hidden = z.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.lstm(hidden)
        decoded = self.output_layer(lstm_out)
        return decoded


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z, x.size(1))
        return decoded, z
