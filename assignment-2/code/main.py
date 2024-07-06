# main.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

from argparse import ArgumentParser 
from nb_data_loader import *
from alt_data_loader import *
from alt_model import ALTModel
from torch.utils.data import DataLoader
from naive_bayes import NaiveBayes
import time
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import pickle

def save_to_file(model, filename):
  path = f"{os.path.dirname(os.path.realpath(__file__))}/models/{filename}.pkl"
  with open(path, 'wb') as f: 
    pickle.dump(model, f)
  f.close()
  print(f"Model saved to {path}")

def import_model(filename):
  path = f"{os.path.dirname(os.path.realpath(__file__))}/models/{filename}.pkl"
  with open(path, 'rb') as f:
    model = pickle.load(f)
  f.close()
  print("Model imported from {path}")
  return model

def nb():
  file_path = f"{os.path.dirname(os.path.realpath(__file__))}/models/naive_bayes.pkl"

  nb_loader = NBDataLoader(args.data_dir)
  x_train, y_train = nb_loader.get_train_data()
  x_val, y_val = nb_loader.get_val_data()
  x_test, y_test = nb_loader.get_test_data()

  # convert data to torch tensor
  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.int64)
  x_val = torch.tensor(x_val, dtype=torch.float32)
  y_val = torch.tensor(y_val, dtype=torch.int64)
  x_test = torch.tensor(x_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.int64)

  # flatten and binarise data
  x_train = (x_train.view(x_train.size(0), -1) > 127).int()
  x_val = (x_val.view(x_val.size(0), -1) > 127).int()
  x_test = (x_test.view(x_test.size(0), -1) > 127).int()

  if not os.path.isfile(file_path):
    print("Model does not exists...\n Creating model.")
    nb = NaiveBayes()
    class_prob = nb.calculate_class_probs(y_train)
    feature_prob = nb.calculate_feature_probs(x_train, y_train)
    nb.train(x_train, y_train)
    save_to_file(nb, "naive_bayes")

  else: 
    print("Model already exists. Skipping model creation.")
    nb = import_model("naive_bayes")
    
    # predict on validation set 
    y_val_pred = nb.predict(x_val)
    val_acc = torch.mean((y_val_pred == y_val).float())
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

  # # predict on test set 
  # y_test_pred = nb.predict(x_test)
  # test_acc = torch.mean((y_test_pred == y_test).float())
  # print(f"Test Accuracy: {test_acc * 100:.2f}%")




USAGE_STRING = """
  USAGE:      python main.py <options>
  EXAMPLES:   (1) python main.py --c nb --d digitdata --mode train
                  - trains the naive bayes classifier on the digit dataset
              (2) python main.py --classifier alt  --data_dir digitdata --mode train --batch_size 64 --epoch 5 --learning_rate 0.0001
                  - trains the alternative model
                  """
  

if __name__ == "__main__":
  parser = ArgumentParser(USAGE_STRING)
  parser.add_argument('-c', '--classifier', help='The type of classifier', choices=['nb', 'alt'], required=True)
  parser.add_argument('-d', '--data_dir', help='the dataset folder name', type=str, required=True)
  parser.add_argument('-m', '--mode', help='train, val or test', type=str, required=True)
  parser.add_argument('-b', '--batch_size', help='batch size', type=int)
  parser.add_argument('-e', '--epoch', help='number of epochs', type=int)
  parser.add_argument('-l', '--learning_rate', help='learning rate', type=float)
  args = parser.parse_args()

  print("Doing classification")
  print("--------------------")
  print("classifier:\t" + args.classifier)

  if args.classifier == "nb":
    nb()
  else:
    """
    choose the alternative model
    """



     

     



