import os
import cv2
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
import pickle

np.random.seed(111)

def raw_to_data_frame(input_path):
    cats = os.listdir(input_path)
    print("Total number of sub-directories found: ", len(cats))
    # Store the meta-data in a dataframe for convinience
    data = []
    for folder in cats:
        new_dir = Path(input_path) / folder
        images = sorted(new_dir.glob('*.jpg'))
        annotations = sorted(new_dir.glob('*.cat'))
        n = len(images)
        for i in range(n):
            img = str(images[i])
            annotation = str(annotations[i])
            data.append((img, annotation))
        print("Processed: ", folder)
    print(" ")

    df = pd.DataFrame(data=data, columns=['img_path', 'annotation_path'], index=None)
    print("Total number of samples in the dataset: ", len(df))
    return df

def load_cats_data():
    df = pickle.load(open("preprocessed/cats.p", "rb"))
    return df

if __name__=="__main__":
    # Define some Paths
    input_path = 'D:\proj\SnapCat\data\/raw'
    df = load_cats_data()
    f, ax = plt.subplots(3, 2, figsize=(20, 15))

    # Get six random samples
    samples = df.sample(6).reset_index(drop=True)

    for i, sample in enumerate(samples.values):
        # Get the image path
        sample_img = df['img_path'][i]
        # Get the annotation path
        sample_annot = df['annotation_path'][i]
        # Read the annotation file
        f = open(sample_annot)
        points = f.read().split(' ')
        points = [int(x) for x in points if x != '']
        # Get the list of x and y coordinates
        xpoints = points[1:19:2]
        ypoints = points[2:19:2]
        # close the file
        f.close()

        ax[i // 2, i % 2].imshow(imread(sample_img))
        ax[i // 2, i % 2].axis('off')
        ax[i // 2, i % 2].scatter(xpoints, ypoints, c='g')

    plt.show()