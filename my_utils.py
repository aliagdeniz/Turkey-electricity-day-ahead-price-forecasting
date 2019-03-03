from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.eager import context
import tensorflow as tf
import numpy as np
import os

def whichMonth(liste):
    year, month, day = int(liste[2]), int(liste[1]), int(liste[0])
    offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    week   = ['Pazar', 
              'Pazartesi', 
              'Salı', 
              'Çarşamba', 
              'Perşembe',  
              'Cuma', 
              'Cumartesi']
    afterFeb = 1
    if month > 2: afterFeb = 0
    aux = year - 1700 - afterFeb
    # dayOfWeek for 1700/1/1 = 5, Friday
    dayOfWeek  = 5
    # partial sum of days betweem current date and 1700/1/1
    dayOfWeek += (aux + afterFeb) * 365                  
    # leap year correction    
    dayOfWeek += aux / 4 - aux / 100 + (aux + 100) / 400     
    # sum monthly and day offsets
    dayOfWeek += offset[month - 1] + (day - 1)               
    dayOfWeek = int(dayOfWeek)% 7
    return week[dayOfWeek]

def weekDay(liste):
    year, month, day = int(liste[2]), int(liste[1]), int(liste[0])
    offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    week   = ['Pazar', 
              'Pazartesi', 
              'Sali', 
              'Çarşamba', 
              'Persembe',  
              'Cuma', 
              'Cumartesi']
    afterFeb = 1
    if month > 2: afterFeb = 0
    aux = year - 1700 - afterFeb
    # dayOfWeek for 1700/1/1 = 5, Friday
    dayOfWeek  = 5
    # partial sum of days betweem current date and 1700/1/1
    dayOfWeek += (aux + afterFeb) * 365                  
    # leap year correction    
    dayOfWeek += aux / 4 - aux / 100 + (aux + 100) / 400     
    # sum monthly and day offsets
    dayOfWeek += offset[month - 1] + (day - 1)               
    dayOfWeek = int(dayOfWeek+1)% 7
    return week[dayOfWeek]

def weekDayNum(gun):
    week   = {'Pazartesi':1, 
              'Sali':2, 
              'Çarşamba':3, 
              'Persembe':4,  
              'Cuma':5, 
              'Cumartesi':6,
              'Pazar':7 }
    return week[gun]

def floataCevir(df):
    df = df.str.replace(',', '?')
    df = df.str.replace('.', '')
    df = df.str.replace('?', '.')
    return df


def adj_r_squared(y, yhat):
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adj_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return adj_r_squared

def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
#     if y_true.any() < 1e-07:
#         print("evet var")
    return np.mean(np.abs((y_true - y_pred) /  y_true )) * 100

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple500_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
