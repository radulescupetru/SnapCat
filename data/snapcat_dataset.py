import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
import torch


transform_image = transforms.Compose([
    # transforms.Resize(256),
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
        return back,offset

    def __init__(self, inputs,targets, img_dir, transform=None):
        self.imgs = []
        self.img_dir = img_dir
        self.transform = transform
        self.size = 512, 512
        self.imgs = pickle.load(open("data/preprocessed/{}.p".format(inputs), "rb"))[:5]
        self.labels = pickle.load(open("data/preprocessed/{}.p".format(targets),"rb"))[:5]
        self.scaler = pickle.load(open("D:\proj\SnapCat\data\preprocessed\scaler.p","rb"))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read image
        image = self.imgs[idx]
        im = Image.open("{}".format(image))
        label = self.labels[idx]

        # Resize image and mentain aspect ratio
        im,offset = self.resize_img(im)
        im = transform_image(im)
        # im = transform_image(im)
        if offset[1] < offset[0]:
            label = [w-offset[1] if i % 2 else w+offset[0] for i, w in enumerate(label)]
        else:
            label = [w + offset[1] if i % 2 else w + offset[0] for i, w in enumerate(label)]
        # Get tensor values between 0-255
        im = im * 255
        scaled_label = np.array(self.scaler.transform(np.array(label).reshape(1,-1)))
        label = torch.from_numpy(scaled_label)
        return im, label.float()
