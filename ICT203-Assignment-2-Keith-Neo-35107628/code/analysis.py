import os
import numpy as np
import matplotlib.pyplot as plt
from alt_data_loader import *
import torch

# update folder path to be dynamic
data_folder = f"{os.path.dirname(os.path.realpath(__file__))}/data/digitdata"

x_train = np.load(os.path.join(data_folder, 'x_train.npy'))
y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
x_val = np.load(os.path.join(data_folder, 'x_val.npy'))
y_val = np.load(os.path.join(data_folder, 'y_val.npy'))
x_test = np.load(os.path.join(data_folder, 'x_test.npy'))
y_test = np.load(os.path.join(data_folder, 'y_test.npy'))


def display_shapes():
    # Display shapes of the data
    print("Shapes of the data:")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)


def visualise_img():
    # Visualise an image
    image_index = 4
    plt.imshow(x_train[image_index], cmap='gray')
    plt.title(f"Label: {y_train[image_index]}")
    plt.show()

def altLoader():
    altD = ALTDataLoader("digitdata", "train")
    print(altD[0])

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.__version__)