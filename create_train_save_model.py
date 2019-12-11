import numpy as np
from tensorflow import keras
import sys
import inspect
import matplotlib.pyplot as plt


def print_network_architecture(file):
    code_lines = list(inspect.getsourcelines(sys._getframe(1))[0])
    index_from = code_lines.index('    # begin\n')
    index_to = code_lines.index('    # end\n')
    print(''.join(code_lines[index_from + 1:index_to]), file=file)


def plot_results(history):
    fig = plt.figure(figsize=(10, 6))
    sub1 = fig.add_subplot(121)
    sub1.set_title('График точности')
    sub1.set_xlabel('Эпоха')
    sub1.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
    sub2 = fig.add_subplot(122)
    sub2.set_title('График функции потерь')
    sub2.set_xlabel('Эпоха')
    sub2.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'])
    plt.tight_layout()
    plt.savefig('graphics/graphic_model_{}.png'.format(num))


def get_network_model(m):
    # begin
    return keras.Sequential([
        keras.layers.Input(m),
        keras.layers.Dense(200, activation='relu',
                           kernel_initializer=keras.initializers.RandomNormal(
                               mean=0.0, stddev=0.1, seed=None)),
        keras.layers.Dense(100, activation='relu',
                           kernel_initializer=keras.initializers.RandomNormal(
                               mean=0.0, stddev=0.1, seed=None)),
        keras.layers.Dense(50, activation='relu',
                           kernel_initializer=keras.initializers.RandomNormal(
                               mean=0.0, stddev=0.1, seed=None)),
        keras.layers.Dense(2, activation='softmax')
    ])
    # end


data = np.loadtxt('train_validation_balanced_data.txt', delimiter=',')
num = 28
optimizer = 'nadam'
epochs = 60
n = len(data)
data_train, data_validation = data[:5 * n // 6, :], data[5 * n // 6:, :]
data_validation_zeros = data_validation[data_validation[:, -1] == 0]
data_validation_ones = data_validation[data_validation[:, -1] == 1]

model = get_network_model(len(data_train[0]) - 2)
model.compile(optimizer=optimizer,
    loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(data_train[:, :-2], data_train[:, -2:], epochs=epochs, verbose=0)

with open('models_histories.txt', 'a', encoding='utf-8') as file_out:
    print('*' * 20, file=file_out)
    print('model' + str(num), file=file_out)
    print_network_architecture(file_out)
    print('optimizer =', optimizer, file=file_out)
    print('epochs =', epochs, file=file_out)
    print('history_acc:', hist.history['accuracy'], file=file_out)
    print('history_loss:', hist.history['loss'], file=file_out)
    print('train_loss: ', hist.history['loss'][-1],
          ', train_acc: ', hist.history['accuracy'][-1], file=file_out)
    loss, acc_ones = model.evaluate(data_validation_ones[:, :-2], data_validation_ones[:, -2:], verbose=0)
    print('validation_ones_loss: ', loss,
          ', validation_ones_acc: ', acc_ones, sep='', file=file_out)

    loss, acc_zeros = model.evaluate(data_validation_zeros[:, :-2], data_validation_zeros[:, -2:], verbose=0)
    print('validation_zeros_loss: ', loss,
          ', validation_zeros_acc: ', acc_zeros, sep='', file=file_out)

    print('total_acc: ', (acc_ones * len(data_validation_ones) +
                          acc_zeros * len(data_validation_zeros)) / len(data_validation),
          file=file_out)
    print('*' * 20, file=file_out)
model.save('models/model{}.h5'.format(num))
plot_results(hist)
