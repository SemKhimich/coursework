import numpy as np
from tensorflow import keras

if __name__ == '__main__':
    model = keras.models.load_model('model_final.h5')
    mean_var = np.loadtxt('mean_var_info.txt', delimiter=',')
    file_names = ['4/data_4.txt', '5/data_5.txt']
    for name in file_names:
        data_test = np.loadtxt(name, delimiter=',')
        data_test[:, :-2] = (data_test[:, :-2] - mean_var[0]) / mean_var[1]**0.5
        data_test_zeros = data_test[data_test[:, -1] == 0]
        data_test_ones = data_test[data_test[:, -1] == 1]
        _, test_acc_zeros = model.evaluate(data_test_zeros[:, :-2], data_test_zeros[:, -2:], verbose=0)
        print('test_acc_zeros: ', test_acc_zeros, sep='')
        _, test_acc_ones = model.evaluate(data_test_ones[:, :-2], data_test_ones[:, -2:], verbose=0)
        print('test_acc_ones: ', test_acc_ones, sep='')
        print('total_acc: ', (test_acc_ones * len(data_test_ones) +
                              test_acc_zeros * len(data_test_zeros)) / len(data_test))
