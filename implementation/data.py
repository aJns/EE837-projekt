import os
import imageio
import numpy as np
from random import shuffle


def load_data(data_path):
    directory = os.fsencode(data_path)
    files = os.listdir(directory)

    filename = os.fsdecode(files[0])
    image = imageio.imread(data_path + "/" + filename)
    im_count = len(files)
    im_shape = image.shape
    data_shape = (im_count,) + im_shape

    x = np.empty(data_shape)
    y = np.empty(im_count)

    shuffle(files) ## mix the list of files up, the aid in training
    for i, file in enumerate(files):
        filename = os.fsdecode(file)
        image = imageio.imread(data_path + "/" + filename)
        x[i,:,:,:] = image
        label = filename[0]
        y[i] = label

    return (x, y)


if __name__ == "__main__":
    (x, y) = load_data("../data/shipsnet")
