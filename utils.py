import tensorflow as tf
from tqdm.notebook import tqdm

# custom progress bar for long training
class tqdm_callback(tf.keras.callbacks.Callback):
    
    def __init__(self, num_epochs, desc, loss=None, val_loss=None):
        self.num_epochs = num_epochs
        self.desc = desc
        self.metrics = {'loss':loss, 'val_loss':val_loss}
    
    def on_train_begin(self, logs={}):
        self.epoch_bar = tqdm(total=self.num_epochs, desc=self.desc)
    
    def on_train_end(self, logs={}):
        self.epoch_bar.close()
        
    def on_epoch_end(self, epoch, logs={}):
        for name in self.metrics:
            self.metrics[name]  = logs.get(name, self.metrics[name])
        self.epoch_bar.set_postfix(mse=self.metrics['loss'], val_mse=self.metrics['val_loss'], refresh=False)
        self.epoch_bar.update()
