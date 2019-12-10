import random
import numpy as np


def parse_dataset(dataset_numbers):
    filename_input = '{0}/description_data_{0}.txt'
    filename_output = '{0}/data_{0}.txt'
    false_results = {'false', 'no'}
    for num in dataset_numbers:
        rows_set = set()
        with open(filename_input.format(num), 'r', encoding='utf-8') as file_input:
            for line in file_input:
                if line[0] not in '0123456789' or '?' in line:
                    continue
                row = line.strip().split(',')
                row[-1:] = ['1', '0'] if row[-1] in false_results else ['0', '1']
                rows_set.add(','.join(row))
        rows_list = list(rows_set)
        random.shuffle(rows_list)
        with open(filename_output.format(num), 'w', encoding='utf-8') as file_output:
            for row in rows_list:
                file_output.write(row + '\n')


def join_datasets(dataset_numbers, filename_output):
    filename_input = '{0}/data_{0}.txt'
    rows_set = set()
    for num in dataset_numbers:
        with open(filename_input.format(num), 'r', encoding='utf-8') as file_input:
            for line in file_input:
                rows_set.add(line)
    rows_list = list(rows_set)
    random.shuffle(rows_list)
    with open(
            filename_output.format(''.join(map(lambda x: str(x), dataset_numbers))),
            'w', encoding='utf-8') as file_output:
        for row in rows_list:
            file_output.write(row)


def balanced_dataset_and_save_mean_var(filename_input, filename_output):
    data = np.loadtxt(filename_input, delimiter=',')
    mean = np.mean(data[:, :-2], axis=0)
    var = np.mean(data[:, :-2], axis=0)
    data[:, :-2] = (data[:, :-2] - mean) / var**0.5
    with open('mean_var_info.txt', 'w', encoding='utf-8') as file:
        print(','.join(map(lambda x: str(x), mean)), file=file)
        print(','.join(map(lambda x: str(x), var)), file=file)
    n = len(data)
    # data, data_validation = data[:4 * n // 5, :], data[4 * n // 5:, :]
    # np.random.shuffle(data_validation)
    # np.savetxt(filename_output_validation, data_validation, delimiter=',')
    data_zeros = data[data[:, -1] == 0]
    data_ones = data[data[:, -1] == 1]
    if len(data_ones) < len(data_zeros):  # balancing classes
        data = np.append(data,
                         data_ones[
                             np.random.randint(0,
                                               len(data_ones) - 1,
                                               len(data_zeros) - len(data_ones))],
                         axis=0)
    else:
        data = np.append(data,
                         data_zeros[
                             np.random.randint(0,
                                               len(data_zeros) - 1,
                                               len(data_ones) - len(data_zeros))],
                         axis=0)
    np.random.shuffle(data)
    np.savetxt(filename_output, data, delimiter=',')


if __name__ == '__main__':
    pass
    # parse_dataset([1, 2, 3, 4, 5])
    # join_datasets([1, 2, 3], 'training_and_validation_data.txt')
    # balanced_dataset('training_and_validation_data.txt',
    #                  'train_validation_balanced_data.txt')
