import random


def create_dataset(dataset_numbers):
    filename_input = '{0}/description_data_{0}.txt'
    rows = set()
    zero_result_set = {'false', 'no'}
    for num in dataset_numbers:
        with open(filename_input.format(num), 'r', encoding='utf-8') as file_input:
            for line in file_input:
                if line[0] not in '0123456789' or '?' in line:
                    continue
                row = line.strip().split(',')
                row[-1:] = ['0', '1'] if row[-1] in zero_result_set else ['1', '0']
                rows.add(','.join(map(lambda x: str(float(x)), row)))
    rows2 = list(rows)
    for i in range(100):
        random.shuffle(rows2)
    with open('dataset3.txt', 'w', encoding='utf-8') as file_output:
        for row in rows2:
            file_output.write(row + '\n')


create_dataset([1, 2, 3, 4, 5])
