Курсовой проект 3 курс 5 семестр  
Тема: Предсказание дефектов в программном продукте по метрикам кода  
Описание файлов:  
train_validation_balanced_data.txt - тренировочный и валидационный датасеты  
4/data_4.txt, 5/data_5.txt - тестирующие датасеты  
mean_var_info.txt - информация о мо и дисперсий параметров тренирочных
и валидационных датасетов  
logs.txt - в файле находится информация о тренировке
на различных конфигурациях сети  
папка graphics - графики точности и функций ошибок от эпохи
на каждой конфигурации из logs.txt  
data_preparation.py - обработка датасетов  
train_validation.py - тренировка нейронной сети  
testing.py - тестирование сети  
my_model.h5 - нейронная сеть  
  
Для запуска тестирования сети необходимо запустить файл testing.py,
который загружает сеть из файла my_model.h5 и тестирует его
на тестирующих датасетах, нормализованных с помощью мо и дисперсий
из файла mean_var_info.txt
