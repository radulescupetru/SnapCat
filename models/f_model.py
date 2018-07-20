import torch
import torch.nn as nn
import torch.nn.functional as F


class PtrNET(nn.Module):
    def __init__(self):
        super(PtrNET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(512 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 18)

    def forward(self, x):
        # first layer [3,256,256]
        x = F.relu(self.conv1(x))   #[64,256,256]
        x = self.maxpool1(F.relu(self.conv2(x)))    #[64,128,128]

        # second layer
        x = F.relu(self.conv3(x))   #[128,128,128]
        x = self.maxpool1(F.relu(self.conv4(x)))    #[128,64,64]

        # third layer
        x = F.relu(self.conv5(x))   #[256,64,64]
        x = F.relu(self.conv6(x))   #[256,64,64]
        x_sec_resid = self.maxpool1(F.relu(self.conv7(x)))  #[256,32,32]

        # fourth layer
        x = F.relu(self.conv8(x_sec_resid)) #[512,32,32]
        x = F.relu(self.conv9(x))   #[512,32,32]
        x_resid = self.maxpool1(F.relu(self.conv10(x))) #[512,16,16]

        # fifth layer
        x = F.relu(self.conv11(x_resid))    #[512,16,16]
        x = F.relu(self.conv12(x))  #[512,16,16]
        x = self.maxpool1(F.relu(self.conv13(x)))   #[512,8,8]

        x =x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))


        return x.float()

