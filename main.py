import numpy as np
from tensorflow import keras
import sys
import inspect

filename = 'dataset_nov_22_all.txt'
array = np.loadtxt(filename, delimiter=',')
file = open('logs_nov_22_all.txt', 'a', encoding='utf-8')
sys.stdout = file
print('**************************************************')
print('**************************************************')
code_lines = list(inspect.getsourcelines(sys._getframe(0))[0])
index_from = code_lines.index('# begin\n')
index_to = code_lines.index('# end\n')
print(''.join(code_lines[index_from + 1:index_to]))
n = len(array)
for i in range(len(array[0]) - 2):  # заполняю нулевые значения средним в столбце
    column = array[:, i:i + 1]
    mean_in_column = column[column != 0].mean()
    column[column == 0] = mean_in_column
for i in range(len(array[0]) - 2):  # нормализация
    max_in_column = array[:, i].max()
    min_in_column = array[:, i].min()
    array[:, i] = (array[:, i] - min_in_column) / (max_in_column - min_in_column)
xy_train, xy_test = array[:5 * n // 6, :], array[5 * n // 6:, :]
xy_test_zeros = xy_test[xy_test[:, -1] == 0]  # одна тестовая выборка один признак
xy_test_ones = xy_test[xy_test[:, -1] == 1]  # вторая другой
# begin
model = keras.Sequential([
        keras.layers.Input(21),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')  # tanh, sigmoid, hard_sigmoid, softsign, relu
    ])
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # sparse_categorical_crossentropy, binary_crossentropy, mean_squared_error
              metrics=['accuracy'])
model.fit(xy_train[:, :21], xy_train[:, 21:], epochs=60, verbose=2)
test_loss_zeros, test_acc_zeros = model.evaluate(xy_test_zeros[:, :21], xy_test_zeros[:, 21:])
test_loss_ones, test_acc_ones = model.evaluate(xy_test_ones[:, :21], xy_test_ones[:, 21:])
# end
print('test_loss_zeros:', test_loss_zeros, '- test_accuracy_zeros:', test_acc_zeros)
print('test_loss_ones:', test_loss_ones, '- test_accuracy_ones:', test_acc_ones)
print('total_accuracy:', (test_acc_zeros * len(xy_test_zeros) + test_acc_ones * len(xy_test_ones)) /
      (len(xy_test_zeros) + len(xy_test_ones)))
file.close()
