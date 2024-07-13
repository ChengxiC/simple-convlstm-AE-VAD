import torch
from torch import nn
from torch.nn import functional as F
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class convlstm(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(convlstm, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv_x = nn.Conv2d(input_channels, hidden_channels*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels*4, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, hidden):
        """

        :param x: [b, ch, h, w], input data
        :param hidden: (h_t, c_t), the previous output hidden h_t, the memory state of the previous cell, c_t
        :return: (h_t, c_t)
        """
        (h_t, c_t) = hidden
        gates = self.conv_x(x) + self.conv_h(h_t)
        i, f, g, o = torch.chunk(gates, chunks=4, dim=1)
        c_t = c_t*f + i*g
        h_t = torch.tanh(c_t)*o
        return (h_t, c_t)


class my_stae(nn.Module):

    def __init__(self, input_channels):
        super(my_stae, self).__init__()
        self.input_channels = input_channels

        self.conv_layers = nn.Sequential(
            # [b, ch, 227, 227] => [b, 128, 55, 55]
            nn.Conv2d(input_channels, 128, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            # [b, 128, 55, 55] => [b, 64, 26, 26]
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=0)
        )

        self.convlstm1 = convlstm(64, 16)
        self.convlstm2 = convlstm(16, 16)
        self.convlstm3 = convlstm(16, 64)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, input_channels, kernel_size=11, stride=4, padding=0)
        )

    # init for convlstm
    def init_lstm_state(self, batch_sz, hidden_channels, h, w):
        h_t = torch.zeros(batch_sz, hidden_channels, h, w).to(device)
        c_t = torch.zeros(batch_sz, hidden_channels, h, w).to(device)
        return (h_t, c_t)

    def forward(self, input, hidden=None):
        """

        :param x: [b, seq, ch, h, w]
        :param hidden: (h_t, c_t), at the beginning, hidden is None
        :return: [b, seq, ch, h, w]
        """
        (batch_sz, seq_len, channels, h, w) = input.shape
        # turn 5-dim tensor to 4-dim tensor for conv
        x = input.reshape(batch_sz*seq_len, channels, h, w)
        x = self.conv_layers(x)
        # turn to 5 dims for conv lstm
        x = torch.reshape(x, [batch_sz, seq_len, 64, 26, 26])

        seq_out = []
        if hidden is None:
            # we do initialization
            h_t1, c_t1 = self.init_lstm_state(batch_sz, 16, 26, 26)
            h_t2, c_t2 = self.init_lstm_state(batch_sz, 16, 26, 26)
            h_t3, c_t3 = self.init_lstm_state(batch_sz, 64, 26, 26)  # to match channels

        for t in range(seq_len):
            x_t = x[:, t, ...]
            h_t1, c_t1 = self.convlstm1(x_t, (h_t1, c_t1))
            h_t2, c_t2 = self.convlstm2(h_t1, (h_t2, c_t2))
            h_t3, c_t3 = self.convlstm3(h_t2, (h_t3, c_t3))
            seq_out.append(h_t3)

        # now we have a list of tensors, then we stack to get 5 dims
        x = torch.stack(seq_out, dim=1)
        # we reshape for deconv
        x = torch.reshape(x, [batch_sz*seq_len, 64, 26, 26])
        x = self.deconv_layers(x)

        x = F.sigmoid(x)

        # we reshape for output
        x = x.reshape(batch_sz, seq_len, channels, h, w)
        return x


"""
start = time.perf_counter()
temp = torch.rand(2, 10, 3, 227, 227).to(device)
model = my_stae(3).to(device)
print(model)
out = model(temp)
print(out.shape)
end = time.perf_counter()
print(f'the processing time is:', end-start, 's')

"""
