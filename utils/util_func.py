import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms

to_pil = transforms.Compose([
    transforms.ToPILImage()
])

def plot_image(image,label):
    scaler = pickle.load(open("data/preprocessed/scaler.p", "rb"))
    # label = scaler.inverse_transform(label)

    xpoints = label[0:18:2]
    ypoints = label[1:18:2]
    # image = np.array(flip(image))
    image = to_pil(image)
    plt.imshow(image)
    plt.scatter(xpoints,ypoints,c="g")
    plt.show()