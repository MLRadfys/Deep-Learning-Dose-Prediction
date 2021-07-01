import keras
import os

class CustomSaver(keras.callbacks.Callback):

    """

    Custom model checkpoints, saves model after each epoch

    """

    def __init__(self, data_dir):

        """

        Args:
            data_dir (str): path to save the Model checkpoints.
        
        Returns:
            None

        """

        self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok= True)

    def on_epoch_end(self, epoch, log={}):

        """

        Saves the model after each epoch.

        Args:
            epoch (int): Actual epoch
        
        Returns:
            None
        
        """

        print('actual epoch:{}...'.format(epoch))

        if epoch > 200:
            if epoch % 1 == 0:
                self.model.save(self.data_dir + "/model_epoch{}.hd5".format(epoch))
                        
        else:
            if epoch % 10 == 0:
                self.model.save(self.data_dir + "/model_epoch{}.hd5".format(epoch))


