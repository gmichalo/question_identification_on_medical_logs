import torch.nn as nn
import torch
import torch.nn.functional as F


class CHARCNN(nn.Module):
    def __init__(self, args, class_number, max_sentence, input_channel, output_channel=256, dropout=0.5,
                 linear_size=1024):
        super(CHARCNN, self).__init__()
        self.name = "CHAR_CNN"
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=args.filter_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size=args.filter_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.pool_size)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size=args.filter_sizes[1]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size=args.filter_sizes[1]),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size=args.filter_sizes[1]),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size=args.filter_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.pool_size)
        )

        temp = - args.filter_sizes[0] + 1 - args.pool_size * (args.filter_sizes[0] -1) -(args.pool_size*args.pool_size*4*(args.filter_sizes[1]-1))
        linear_size_temp = int((max_sentence + temp) / (args.pool_size * args.pool_size * args.pool_size)) * output_channel
        self.fc1 = nn.Sequential(
            nn.Linear(linear_size_temp, linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Linear(linear_size, class_number)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
