import torch 
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
import numpy as np

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
    print(f"Confusion matrix saved to {dir_path}/{model_name}_confusion_matrix.png\n")

def plot_roc_curve(y_true, y_scores, model_name, num_classes):
    dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Ensure y_true is a numpy array for proper indexing
    y_true = np.array(y_true)

    # Compute ROC curve and ROC area for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    colors = plt.cm.get_cmap('tab10', num_classes).colors
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(f"{dir_path}/{model_name}_roc_curve.png")
    plt.show()
    print(f"ROC curves saved to {dir_path}/{model_name}_roc_curve.png\n")


def plot_class_wise_metrics(precision, recall, f1_score, model_name, num_classes):
    dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    x = range(num_classes)
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x, precision, width, label="Precision")
    plt.bar([p + width for p in x], recall, width, label="Recall")
    plt.bar([p + width*2 for p in x], f1_score, width, label="F1 Score")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title(f"Class-wise Precision, Recall and F1 Score ({model_name})")
    plt.xticks([p + width for p in x], x)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f"{dir_path}/{model_name}_class_wise_metrics.png")
    print(f"Class-wise performance metrics saved to {dir_path}/{model_name}_Class_wise_metrics.png\n")

def plot_precision_recall_curve(y_true, y_scores, model_name, num_classes):
    dir_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    precision = dict()
    recall = dict()
    average_precision = dict()

    # Ensure y_true is a numpy array for proper indexing
    y_true = np.array(y_true)

    # Compute precision-recall curve and average precision for each class
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
        average_precision[i] = average_precision_score(y_true == i, y_scores[:, i])

    # Plot precision-recall curves
    plt.figure()
    colors = plt.cm.get_cmap('tab10', num_classes).colors
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve (area = %0.2f) for class %d' % (average_precision[i], i))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(f"{dir_path}/{model_name}_precision_recall_curve.png")
    plt.show()
    print(f"Precision-Recall curves saved to {dir_path}/{model_name}_precision_recall_curve.png\n")