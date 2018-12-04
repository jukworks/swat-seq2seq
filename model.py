import torch
import torch.nn.parallel
from torch import nn

import conf


class Encoder(nn.Module):
    def __init__(self, n_inputs, n_hiddens):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.lstm1 = nn.LSTMCell(input_size=n_inputs, hidden_size=n_hiddens)
        self.lstm2 = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)
        self.lstm3 = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)
        self.lstm4 = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)

    def init_hidden_and_cell_state(self, batch_size, dev):
        return (
            torch.zeros((batch_size,
                         self.n_hiddens),
                        dtype=torch.float,
                        device=dev),
            torch.zeros((batch_size,
                         self.n_hiddens),
                        dtype=torch.float,
                        device=dev)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_gpu = torch.device('cuda:{}'.format(x.get_device()))
        hc1 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        hc2 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        hc3 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        hc4 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        x = x.view(batch_size, conf.WINDOW_GIVEN, -1)
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        encoder_outs = []
        for point in x:
            hc1 = self.lstm1(point, hc1)
            hc2 = self.lstm2(hc1[0], hc2)
            hc3 = self.lstm3(hc2[0], hc3)
            hc4 = self.lstm4(hc3[0], hc4)
            encoder_outs.append(hc4[0].unsqueeze(1))
        out = torch.cat(encoder_outs, dim=1)  # (B, seq, hiddens)
        return out, hc4[1]


class AttentionDecoder(nn.Module):
    def __init__(self, n_hiddens, n_features):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.lstm = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)
        self.extend = nn.Linear(n_features, n_hiddens)
        self.attn = nn.Sequential(nn.Linear(n_hiddens * 2, conf.WINDOW_GIVEN), nn.Softmax(dim=1))
        self.attn_combine = nn.Sequential(nn.Linear(n_hiddens * 2, n_hiddens), nn.ReLU())
        self.out = nn.Linear(n_hiddens, n_features)

    def init_cell_state(self, batch_size, dev):
        return torch.zeros((batch_size, self.n_hiddens), dtype=torch.float, device=dev)

    def forward(self, encoder_outs, context, predict):
        batch_size = predict.size(0)
        x_gpu = torch.device('cuda:{}'.format(predict.get_device()))
        predict = predict.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        h_i = context
        c_i = self.init_cell_state(batch_size, x_gpu)
        for idx in range(conf.WINDOW_PREDICT):
            inp = self.extend(predict[idx])
            # input:(B, params -> hiddens), c_i:(B, hiddens) => attn_weights:(B, window-size given)
            attn_weights = self.attn(torch.cat((inp, context), dim=1))
            # BMM of attn_weights (B, 1, windows given) and encoding outputs (B, seq, hiddens) -> (B, 1, hiddens)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outs)
            attn_combine = self.attn_combine(torch.cat((inp, attn_applied.squeeze(1)), dim=1))
            h_i, c_i = self.lstm(attn_combine, (h_i, c_i))
            # h_i : (B, hiddens), self.out : (B, hiddens) -> (B, # of features)
        return self.out(h_i)  # (B, n_outs, # of features)


def main():
    print(Encoder(10, conf.N_HIDDEN_CELLS))
    print(AttentionDecoder(conf.N_HIDDEN_CELLS, 10))


if __name__ == '__main__':
    main()
