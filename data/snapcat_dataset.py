import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
import torch


transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

transform_label = transforms.Compose([
    transforms.ToTensor()
])


class SnapCatDataset(data.Dataset):

    def resize_img(self, image, method=Image.ANTIALIAS):
        max_size = self.size
        image.thumbnail(max_size, method)
        offset = (int((max_size[0] - image.size[0]) / 2), int((max_size[1] - image.size[1]) / 2))
        back = Image.new("RGB", max_size, "black")
        back.paste(image, offset)
        return back

    def __init__(self, inputs,targets, img_dir, transform=None):
        self.imgs = []
        self.img_dir = img_dir
        self.transform = transform
        self.size = 512, 512
        self.imgs = pickle.load(open("data/preprocessed/{}.p".format(inputs), "rb"))[:10]
        self.labels = pickle.load(open("data/preprocessed/{}.p".format(targets),"rb"))[:10]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read image
        image = self.imgs[idx]
        im = Image.open("{}/{}".format(self.img_dir, image))
        label = self.labels[idx]

        # Resize image and mentain aspect ratio
        im = transform_image(self.resize_img(im))

        # Get tensor values between 0-255
        im = im * 255
        label = torch.from_numpy(np.array(label))
        return im, label.float()
