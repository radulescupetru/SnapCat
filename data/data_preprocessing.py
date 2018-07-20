import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

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


def save_inputs_and_targets(df):
    X = df['img_path']
    y = []
    for sample in df['annotation_path']:
        # Read the annotation file
        f = open(sample)
        points = f.read().split(' ')
        points = [int(x) for x in points if x != '']
        y.append(points[1:])
        f.close()
    pickle.dump(X, open("preprocessed/inputs.p", "wb"))
    pickle.dump(y, open("preprocessed/targets.p", "wb"))
    return X,y


if __name__=="__main__":
    # Define some Paths
    input_path = '/home/ddinu/PycharmProjects/SnapCat/data/raw/cats'
    df = raw_to_data_frame(input_path)
    # df = load_cats_data()
    save_inputs_and_targets(df)
