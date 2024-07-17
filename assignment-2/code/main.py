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
from torchmetrics import Accuracy
import pickle
from sklearn.model_selection import ParameterGrid
from utils import *

def nb(args):
  """
  Trains or tests the Naive Bayes model based on the provided arguments. 
  
  Parameters: 
  args (Namespace): The command-line arguments.
  """
  nb_loader = NBDataLoader(args.data_dir)
  x_train, y_train, x_val, y_val, x_test, y_test = process_data(nb_loader)
  
  if args.mode == "train":
    best_smoothing = 0
    best_accuracy = 0
    for sf in [0.1, 0.5, 1.0, 1.5, 2.0]:
      print(f"Testing smoothing factor: {sf}")
      nb = NaiveBayes(smoothing_factor=sf)
      nb.train(x_train, y_train)
      val_accuracy = evaluate_model(nb, x_val, y_val, "Validation")
      if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_smoothing = sf
    
    print(f"Best smoothing factor: {best_smoothing} with accuracy: {best_accuracy * 100:.2f}%")
    nb = NaiveBayes(smoothing_factor=best_smoothing)
    nb.train(x_train, y_train)
    
    save_to_file(nb, "naive_bayes")
    print("Run command with mode=test to test the model.")
    
  elif args.mode == "test":
    try:
      nb = import_model("naive_bayes")
    except FileNotFoundError:


USAGE_STRING = """
  USAGE:      python main.py <options>
  EXAMPLES:   (1) python main.py --c nb --d digitdata --mode train
                  - trains the naive bayes classifier on the digit dataset
              (2) python main.py --classifier alt  --data_dir digitdata --mode train 
                  - trains the alternative model
              (3) Python main.py --compare --d digitdata 
                  """
  

if __name__ == "__main__":
  parser = ArgumentParser(USAGE_STRING)
  parser.add_argument('-c', '--classifier', help='The type of classifier', choices=['nb', 'alt'], required=False)
  parser.add_argument('-d', '--data_dir', help='the dataset folder name', type=str, required=True)
  parser.add_argument('-m', '--mode', help='train, val or test', type=str, required=False)
  parser.add_argument('--compare', action='store_true', help='Compare the Naive Bayes and CNN models')
  args = parser.parse_args()

  print("Doing classification")
  print("--------------------")
  print("classifier:\t" + str(args.classifier))

  if not args.compare:
    if args.classifier == "nb":
      nb(args)
    else:
      alt_cnn(args)
  else:
    compare_models(args)



     

     



