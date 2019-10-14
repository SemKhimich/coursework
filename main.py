import numpy as np
from tensorflow import keras
import sys
import inspect

filename = 'dataset2.txt'
array = np.loadtxt(filename, delimiter=',')
n = len(array)
x_train, y_train, x_test, y_test = \
    array[:5 * n // 6, :-2], array[:5 * n // 6, -2:], \
    array[5 * n // 6:, :-2], array[5 * n // 6:, -2:]
for i in range(len(x_train[0])):  # нормализация
    max_in_column = max(x_train[:, i].max(), x_test[:, i].max())
    min_in_column = min(x_train[:, i].min(), x_test[:, i].min())
    x_train[:, i] = (x_train[:, i] - min_in_column) / (max_in_column - min_in_column)
    x_test[:, i] = (x_test[:, i] - min_in_column) / (max_in_column - min_in_column)

# d = {}
# for line in array:
#     d.setdefault(line[-2], 0)
#     d[line[-2]] += 1
# for key in d:
#     print('key = ', key, ': ', d[key] / len(array))

file = open('logs2.txt', 'a', encoding='utf-8')
sys.stdout = file
print('**************************************************')
print('**************************************************')
code_lines = list(inspect.getsourcelines(sys._getframe(0))[0])
index_from = code_lines.index('# begin\n')
index_to = code_lines.index('# end\n')
print(''.join(code_lines[index_from + 1:index_to]))

# begin
model = keras.Sequential([
    keras.layers.Input(21),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # tanh, sigmoid, hard_sigmoid, softsign, relu
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # sparse_categorical_crossentropy, binary_crossentropy, mean_squared_error
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
# end

test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_loss:', test_loss, '- test_accuracy:', test_acc)
file.close()
