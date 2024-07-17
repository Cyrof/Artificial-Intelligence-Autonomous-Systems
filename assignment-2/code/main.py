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

def prepare_data(x, y):
  """
  Prepares data for training/testing by converting to torch tensors and binarising. 
  
  Parameters: 
  x (ndarray): Input data.
  y (ndarray): Labels.
  
  Returns: 
  Tuple[torch.Tensor, torch.Tensor]: Processed input data and labels.
  """
  x = torch.tensor(x, dtype=torch.float32)
  y = torch.tensor(y, dtype=torch.int64)
  x = (x.view(x.size(0), -1) > 127).int()
  return x, y

def process_data(dataloader):
  """
  Processes the data using the given data loader. 
  
  Parameters: 
  dataloader (NBDataLoader): The data loader to use for loading the data.
  
  Returns: 
  Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  Training, validation, and test data and labels.
  """
  x_train, y_train = dataloader.get_train_data()
  x_val, y_val = dataloader.get_val_data()
  x_test, y_test = dataloader.get_test_data()
  
  x_train, y_train = prepare_data(x_train, y_train)
  x_val, y_val = prepare_data(x_val, y_val)
  x_test, y_test = prepare_data(x_test, y_test)

  return x_train, y_train, x_val, y_val, x_test, y_test

def evaluate_model(model, x, y, dataset_name):
  y_pred = model.predict(x)
  accuracy = torch.mean((y_pred == y).float())
  print(f"\n{dataset_name} Accuracy: {accuracy * 100:.2f}%")
  return accuracy

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
      # val_accuracy, _, _, _, _ = nb.test_model(x_val, y_val, "val")
      val_accuracy = evaluate_model(nb, x_val, y_val, "Validation")

      if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_smoothing = sf
    
    print(f"Best smoothing factor: {best_smoothing} with accuracy: {best_accuracy * 100:.2f}%")
    nb = NaiveBayes(smoothing_factor=best_smoothing)
    nb.train(x_train, y_train)
    
    save_to_file(nb, "naive_bayes")
    
  elif args.mode == "test":
    try:
      nb = import_model("naive_bayes")
    except FileNotFoundError:
      raise RuntimeError("No Naive bayes model trained. Please run with `--mode train` first.")
    nb.test_model(x_val, y_val, "Validation")
    nb.test_model(x_test, y_test, "Test")
    print(f"The predicted values for validation: {nb.predict(x_val)}")
    print(f"The predicted values for test: {nb.predict(x_test)}")
  else: 
    raise ValueError("Unknown argument used for --mode.")

def alt_cnn(args):
  alt_loader = ALTDataLoader(args.data_dir, args.mode)
  criterion = nn.CrossEntropyLoss()
  accuracy = Accuracy(task="multiclass", num_classes=10)

  if args.mode == "train":
    prompt_str = """
      Training CNN model

      Choose an option: 
      (1) User predifined best hyperparameters
      (2) Perform dynamic hyperparameter tuning

      Note: 
      Option (1) will train the model using a set of predifined hyperparameters known to work well.
      Option (2) will perform a thorough search over multiple combinations of hyperparameters to find the best ones.
      WARNING: Dynamic hyperparameter tuning can take a significant amount of time depending on your system's capabilities.

      Please enter 1 or 2:
    """
    
    user_choice = input(prompt_str)

    if user_choice == '2':
      param_grid = {
        'batch_size': [32, 64, 128],
        'learning_rate': [0.001, 0.01, 0.1],
        'epoch': [8, 16, 32, 64]
      }
      best_acc = 0
      best_params = {}

      param_combinations = list(ParameterGrid(param_grid))
      print(f"Total parameter combinations to try: {len(param_combinations)}")

      for params in tqdm(ParameterGrid(param_grid), desc="Grid Search"):
        print(f"\nTrying parameters: {params}")
        train_dataloader = DataLoader(alt_loader, batch_size=params['batch_size'], shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
        test_dataloader = DataLoader(ALTDataLoader(args.data_dir, "val"), batch_size=params['batch_size'], shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
        cnn = ALTModel()
        optimiser = optim.Adam(cnn.parameters(), lr=params['learning_rate'])
        cnn.train_model(train_dataloader, test_dataloader, epochs=params['epoch'], criterion=criterion, optimiser=optimiser, accuracy=accuracy)

        current_acc = cnn.test_model(test_dataloader, criterion, accuracy)
        print(f"Current accuracy: {current_acc:.5f}")

        if current_acc > best_acc:
          best_acc = current_acc
          best_params = params
          save_to_file(cnn, 'cnn')
      
      print(f"Best parameters: {best_params}")
      print(f"Best accuracy: {best_acc}")
    
    elif user_choice == '1':
      train_dataloader = DataLoader(alt_loader, batch_size=64, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
      test_dataloader = DataLoader(ALTDataLoader(args.data_dir, "val"), batch_size=64, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
      cnn = ALTModel()
      optimiser = optim.Adam(cnn.parameters(), lr=0.001)
      cnn.train_model(train_dataloader, test_dataloader, 15, criterion, optimiser, accuracy)
      
      save_to_file(cnn, 'cnn')
    
  elif args.mode =="test":
    test_dataloader = DataLoader(alt_loader, batch_size=32, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2)
    try:
      cnn = import_model('cnn')
    except FileNotFoundError:
      raise RuntimeError("No CNN model trained. Please run with `--mode train` first.")
    cnn.test_model(test_dataloader, criterion, accuracy)
    predicted = cnn.predict(test_dataloader)
    print(f"Predicted classes: {predicted}")
    
  else: 
    raise ValueError("Unknown argument used for --mode.")
  
def compare_models(args):
  nb_loader = NBDataLoader(args.data_dir)
  x_train, y_train, x_val, y_val, x_test, y_test = process_data(nb_loader)
  try:
    nb = import_model("naive_bayes")
  except FileNotFoundError:
    raise RuntimeError("No Naive bayes model trained. Please run with `--mode train` first.")
  print("Evaluating Naive Bayes Model...")
  nb_acc, nb_f1, nb_conf, nb_precision, nb_recall = nb.test_model(x_test, y_test, "Test")
  plot_confusion_matrix(nb_conf, 'Naive Bayes')
  # plot_precision_recall(nb_precision, nb_recall, "Naive Bayes")

  alt_loader = ALTDataLoader(args.data_dir, "test")
  criterion = nn.CrossEntropyLoss()
  accuracy = Accuracy(task="multiclass", num_classes=10)
  test_dataloader = DataLoader(alt_loader, batch_size=32, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2)
  try:
    cnn = import_model('cnn')
  except FileNotFoundError:
    raise RuntimeError("No CNN model trained. Please run with `--mode train` first.")
  print("\nEvaluating CNN Model...")
  cnn_acc, cnn_f1, cnn_conf, cnn_precision, cnn_recall = cnn.test_model(test_dataloader, criterion, accuracy)
  plot_confusion_matrix(cnn_conf, 'CNN')
  # plot_precision_recall(cnn_precision, cnn_recall, "CNN")

  if nb_acc > cnn_acc:
    print(f"\nNaive bayes performs better on the test set.")
  else:
    print(f"\nCNN performs better on the test set.")



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



     

     

