import random


def create_dataset(dataset_numbers):
    filename_input = '{0}/description_data_{0}.txt'
    rows_set = set()
    zero_result_set = {'false', 'no'}
    for num in dataset_numbers:
        with open(filename_input.format(num), 'r', encoding='utf-8') as file_input:
            for line in file_input:
                if line[0] not in '0123456789' or '?' in line:
                    continue
                row = line.strip().split(',')
                row[-1:] = ['1', '0'] if row[-1] in zero_result_set else ['0', '1']
                rows_set.add(','.join(map(lambda x: str(float(x)), row)))
    rows_list = []
    for row in rows_set:
        rows_list.append(row)
        if row[-3:] == '1.0':
            rows_list.append(row)
            rows_list.append(row)
            rows_list.append(row)
    for i in range(100):
        random.shuffle(rows_list)
    with open('dataset_nov_22_all.txt', 'w', encoding='utf-8') as file_output:
        for row in rows_list:
            file_output.write(row + '\n')


create_dataset([1, 2, 3, 4, 5])
