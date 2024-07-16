import torch 
import os
import pickle

def save_to_file(model, filename):
    dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/models"
    path = f"{dir_path}/{filename}.pkl"

    # create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # save the model to the specified path
    with open(path, 'wb') as f: 
        pickle.dump(model, f)

    print(f"Model saved to {path}")

def import_model(filename):
    path = f"{os.path.dirname(os.path.realpath(__file__))}/models/{filename}.pkl"
    with open(path, 'rb') as f:
        model = pickle.load(f)
    f.close()
    print(f"Model imported from {path}")
    return model    