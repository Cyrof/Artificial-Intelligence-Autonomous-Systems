class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Attributes: 
    patience (int): Number of epochs with no improvement after which training will be stopped.
    min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    counter (int): Counter to keep track of the number of epochs with no improvement.
    best_loss (float or None): Best validation loss achieved so far.
    early_stop (bool): Flag to indicate if early stopping has been triggered.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Initialise the EarlyStopping object.

        Parameters:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if the validation loss has improved.

        Parameters: 
        val_loss (float): Validation loss to be checked.

        Returns: 
        bool: True if the validation loss has improved, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            # Loss increased, increment patience customer
            self.counter += 1
            if self.counter >= self.patience:
                # Reached patience limit, set early_stop flag
                self.early_stop = True
        else: 
            # Loss decreased or stayed the same, reset counters
            self.best_loss = val_loss
            self.counter = 0