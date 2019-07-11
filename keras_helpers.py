import gspread
import numpy as np
import keras.backend as K
import tensorflow as tf

from datetime import datetime as dt
from keras.callbacks import Callback
from oauth2client.service_account import ServiceAccountCredentials

kms = 300
c = 299792.458 #Speed of Light in kms
z_tol = np.sqrt((1+kms/c)/(1-kms/c))-1

def sci_standard(y_t, y_p):
    return K.mean(K.less_equal(np.abs(y_t - y_p), z_tol))

class EndEpochMetric(Callback):
    """
    This callback will evaluate given metrics on given datasets. It will
    place them in the logs dictionary with name given as
    {dataset_name}_custom_{metric_function_name}. If the metric results in a
    tensor it will be evaluated. If it returns anything else nothing will be
    done to the result.

    :param metrics: list of functions to evaluate on datasets.
    :param datasets: list of datasets to evaluate with metrics. dataset must be
    a tuple with first item being name of dataset, next be data, and lastly
    the targets. Example: ("train", train_dataset.flux, traindataset.zs)
    """

    def __init__(self, metrics, datasets, *args, **kwargs):
        super(EndEpochMetric, self).__init__(*args, **kwargs)
        self.metrics = metrics
        self.datasets = datasets

    def on_epoch_end(self, epoch, logs={}):
        for ix, (dataset_name, x, y) in enumerate(self.datasets):
            for cur_metric in self.metrics:
                v = cur_metric(self.model.predict(x), y)
                if isinstance(v, tf.Tensor):
                    v = K.eval(v)
                logs[dataset_name + '_custom_' + cur_metric.__name__] = float(v)

class GoogleSheetsWriter(Callback):
    """
    This module is a keras callback that writes the results of the training to
    a google spreadsheet. It generates a new seperate worksheet on
    initalization of the class.

    This callback logs all the metrics stored in the logs after the epoch
    has ended.

    :param json_keyfile: need a oauth json_keyfile to access Google Drive files.
    :param training_spreadsheet: name of training spreadsheet to add worksheets
    of training information to.
    :param model_info: model used in training. Makes use of keras model.summary method.
    :param exp_info: dict containing configuration information for the
    experiment set up.
    """

    def __init__(self, json_keyfile, training_spreadsheet, model_info, exp_info,
                 *args, **kwargs):
        super(GoogleSheetsWriter, self).__init__(*args, **kwargs)

        # use creds to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile,
                                                                 scope)
        self.gc = gspread.authorize(creds)

        # open the training sheet and make a new worksheet.
        self.spreadsheet = self.gc.open(training_spreadsheet)
        str_format = "%Y-%m-%d-%H_%M_%S"
        self.ws = self.spreadsheet.add_worksheet(
                'TrainingData_{}'.format(dt.now().strftime(str_format)),
                10, 10)

        # writing model info
        self.ws.append_row(['-------------------- Model Info --------------------'])
        model_info.summary(print_fn=lambda x: self.ws.append_row([str(x)]))

        # writing experiment info
        self.ws.append_row(['-------------------- Experiment Info --------------------'])
        self.ws.append_row(list(exp_info.keys()))
        self.ws.append_row(list(exp_info.values()))
        
        self.ws.append_row(['-------------------- Training Results --------------------'])

    def on_epoch_end(self, epoch, logs={}):
        try:
            # writing evaluation metrics.
            if epoch == 0:
                self.ws.append_row(['epoch'] + list(logs.keys()))
            self.ws.append_row([epoch] + list(logs.values()))
        except Exception as e:
            print("Couldnt write to sheet:{}\n {}" .format(['epoch'] + list(logs.keys()), 
                                                           [epoch] + list(logs.values())))
