from selenium.webdriver.common.by import By
import time
from selenium import webdriver
def get_user_profile(username):
    try:
        profile_url = "https://www.instagram.com/" + username + "/"
        driver = webdriver.Chrome()
        driver.get(profile_url)
        time.sleep(15)
        parent_element = driver.find_elements(By.CSS_SELECTOR, '._ac2a')  # 3 개 있음
        if len(parent_element) != 3:
            time.sleep(5)
        else:
            post_count = parent_element[0].find_element(By.CSS_SELECTOR, '.html-span').text
            followers = parent_element[1].find_element(By.CSS_SELECTOR, '.html-span').text
            following = parent_element[2].find_element(By.CSS_SELECTOR, '.html-span').text
            file = open('./data/user_data.csv', 'a', encoding='euc-kr')
            file.write(f"\n{username},{followers},{following},{post_count}")
            file.close()
            return True
    except:
        return False


