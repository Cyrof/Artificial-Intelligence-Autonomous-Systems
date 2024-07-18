import torch 
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, PrecisionRecallDisplay

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

def plot_confusion_matrix(cm, model_name):
    dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(colorbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{dir_path}/{model_name}_confusion_matrix.png")
    print(f"Confusion matrix saved to {dir_path}/{model_name}_confusion_matrix.png")

def plot_precision_recall(precision, recall, model_name):
    dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure()
    disp = PrecisionRecallDisplay(precision, recall)
    # disp.plot(colorbar=False)
    plt.title(f"{model_name} Pricision Recall curve")
    # labels = ['Precision', 'Recall']
    # scores = [precision, recall]

    # plt.plot(recall, precision, marker='o', color='b', label=f'{model_name} Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'{model_name} Precision and Recall Scores')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.grid(True)
    # plt.legend(loc="lower left")
    plt.savefig(f"{dir_path}/{model_name}_precision_recall.png")
    print(f"Precision-Recall curve saved to {dir_path}/{model_name}_precision_recall.png")