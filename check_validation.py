import numpy as np
from tensorflow import keras

model = keras.models.load_model('models/model19.h5')
data = np.loadtxt('train_validation_balanced_data.txt', delimiter=',')
n = len(data)
data_validation = data[5 * n // 6:, :]
data_validation_zeros = data_validation[data_validation[:, -1] == 0]
data_validation_ones = data_validation[data_validation[:, -1] == 1]
loss, acc_ones = model.evaluate(data_validation_ones[:, :-2], data_validation_ones[:, -2:], verbose=0)
print('validation_ones_loss: ', loss,
      ', validation_ones_acc: ', acc_ones, sep='')

loss, acc_zeros = model.evaluate(data_validation_zeros[:, :-2], data_validation_zeros[:, -2:], verbose=0)
print('validation_zeros_loss: ', loss,
      ', validation_zeros_acc: ', acc_zeros, sep='')

print('total_acc: ', (acc_ones * len(data_validation_ones) +
                      acc_zeros * len(data_validation_zeros)) / len(data_validation))
