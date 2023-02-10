from selenium import webdriver
import time
import pickle
from tqdm import tqdm
from selenium.webdriver.common.by import By
def preparing_cite():
    options = webdriver.ChromeOptions()
    a = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.102 Safari/537.36'
    # user-agent
    # options.add_argument("user-agent=Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0")
    options.add_argument(a)
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

    return driver, marks_tidy,marks, count_tidy, city, counts


