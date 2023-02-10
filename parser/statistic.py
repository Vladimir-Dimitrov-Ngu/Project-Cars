import pandas as pd
from tqdm import tqdm
def statistics_info(count_tidy, counts, marks, city, statistics = True) -> pd.DataFrame:
    '''
    Функция собирает статистику по определенному городу
    '''
    if statistics == True:
        for i in tqdm(range(len(count_tidy)), desc='Считаю статистики'):
            counts.append(count_tidy[i].text.split('\n')[0])
        dict_for_statistic = {
            'Marks': marks,
            'Counts': counts,
            'City': [city] * len(marks)
        }
        assert len(marks) == len(counts), 'Ошибка в реализации или в выдаче сайте'
        pd.DataFrame(dict_for_statistic).to_csv(f'Статистика_{city}.csv')
