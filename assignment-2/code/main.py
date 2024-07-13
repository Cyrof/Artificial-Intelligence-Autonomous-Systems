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
from errors import *

def save_to_file(model, filename):
  """
  Saves the m odel to a specified file.
  
  Parameters: 
  model (object): The model to save.
  filename (str): The name of the file to save the model to.
  """
  path = f"{os.path.dirname(os.path.realpath(__file__))}/models/{filename}.pkl"
  with open(path, 'wb') as f: 
    pickle.dump(model, f)
  f.close()
  print(f"Model saved to {path}")

def import_model(filename):
  """
  Imports a model from a specified file.
  
  Parameters:
  filename (str): The name of the file to import the model from.
  
  Returns: 
  object: The imported model.
  """
  path = f"{os.path.dirname(os.path.realpath(__file__))}/models/{filename}.pkl"
  with open(path, 'rb') as f:
    model = pickle.load(f)
  f.close()
  print(f"Model imported from {path}")
  return model

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
  """
  Evaluates the model on a given dataset.
  
  Parameters: 
  model (object): The model to evaluate.
  x (torch.Tensor): The input data.
  y (torch.Tensor): The labels.
  dataset_name (str): The name of the dataset.
  """
  y_pred = model.predict(x)
  print(y_pred)
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

def calculate_acc(model, dataloader):
  model.eval()
  predicted_values = model.predict(dataloader)
  correct = 0
  total = 0
  idx = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
      batch_size = labels.size(0)
      preds = predicted_values[idx:idx + batch_size]
      idx += batch_size
      total += batch_size
      correct += (torch.tensor(preds) == labels.cpu()).sum().item()
  acc = 100 * correct / total 
  print(f"Accuracy of the network on the test images: {acc:.2f}%")
  
def alt_cnn(args):
  alt_loader = ALTDataLoader(args.data_dir, args.mode)
  
  if args.mode == "train":
    train_dataloader = DataLoader(alt_loader, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ALTDataLoader(args.data_dir, "val"), batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(alt_loader[1], batch_size=args.batch_size, shuffle=True)
    cnn = ALTModel()
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(), lr=args.learning_rate)
    accuracy = Accuracy(task="multiclass", num_classes=10)
    cnn.train_model(train_dataloader, test_dataloader, 15, criterion, optimiser, accuracy)
    
    save_to_file(cnn, 'cnn')
    
  elif args.mode =="test":
    dataloader = DataLoader(alt_loader, batch_size=args.batch_size, shuffle=False)
    cnn = import_model('cnn')
    calculate_acc(cnn, dataloader)
    
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
    alt_cnn(args)



     

     



