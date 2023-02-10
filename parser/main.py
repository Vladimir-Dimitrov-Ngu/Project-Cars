from selenium import webdriver
import time
import datetime
from selenium.webdriver.common.by import By
import pickle
from tqdm import tqdm
import pandas as pd

# agents options
statistics = True


# options
options = webdriver.ChromeOptions()

# user-agent
options.add_argument("user-agent=Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0")

# for ChromeDriver version 79.0.3945.16 or over
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(executable_path=r'C:\Users\user\Project-Cars\drivers\Chromedriver\chromedriver.exe',
                          options=options)

cite = 'https://auto.ru/'


driver.get(cite)
driver.find_element(by = By.XPATH, value = '//*[@id="root"]/div/div/form/div[2]/div/div/div[1]/input').click()
time.sleep(15)
for cookie in pickle.load(open(f"auto_cookies", 'rb')):
    driver.add_cookie(cookie)
driver.refresh()
marks_tidy = driver.find_elements(by = By.CLASS_NAME, value = 'IndexMarks__item')
count_tidy = driver.find_elements(by = By.CLASS_NAME, value = 'IndexMarks__item-count')
marks = []
counts = []
city = driver.find_element(by = By.CLASS_NAME, value = 'GeoSelect__titleShrinker-wjCdV').text.split(' ')[0]
for j in tqdm(range(len(marks_tidy)), desc = 'Работаю с марками'):
    marks.append(marks_tidy[j].text.split('\n')[0])

if statistics == True:
    for i in tqdm(range(len(count_tidy)), desc = 'Считаю статистики'):
        counts.append(count_tidy[i].text.split('\n')[0])
    d = {
        'Marks': marks,
        'Counts': counts,
        'City': [city] * len(marks)
    }
    assert len(marks) == len(counts), 'Ошибка в реализации или в выдаче сайте'
    pd.DataFrame(d).to_csv(f'Статистика_{city}.csv')

urls_all = pd.DataFrame()
for mark in tqdm(marks_tidy, desc = 'Парсирую юрл марок (вообще)'):
    print(f'Сейчас работаю с маркой {mark}')
    mark.click()
    link = []
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    page_max = driver.find_element(by = By.XPATH, value = '//*[@id="lmcQaMxB6xjLkJq45u4ay"]/div[10]/div/span')\
        .text.split('\n')[-1]
    print(f'Всего будет {page_max} страниц')
    for i in tqdm(range(2, int(page_max) + 1), desc = f'page for {mark}'):
        urls = driver.find_elements(by = By.CLASS_NAME, value = 'LazyImage__image')
        for url in urls:
            link.append(url.get_attribute('src'))
        driver.find_element(by = By.LINK_TEXT, value = f'{i}').click()
        if page_max == 1:
            break
    d = {'Urls' : link,
         'Mark' : [mark] * len(link)
         }
    urls_all.append(d)
urls_all.to_csv('Urls.csv')