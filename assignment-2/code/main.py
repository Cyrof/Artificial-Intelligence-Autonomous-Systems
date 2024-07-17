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
      raise RuntimeError("No Naive bayes model trained. Please run with mode=train first.")
    evaluate_model(nb, x_val, y_val, "Validation")
    evaluate_model(nb, x_test, y_test, "Test")
  else: 
    raise UnknownArgs("Unknown argument used for --mode.")
  

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
    nb(args)
  else:
    """
    choose the alternative model
    """



     

     



