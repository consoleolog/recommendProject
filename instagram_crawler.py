from selenium import webdriver
from selenium.webdriver.common.by import By
from multiprocessing.dummy import Pool as ThreadPool
from save_post import get_post_data
import time
import os
import pandas as pd
import numpy as np
from do_login import do_login
from save_user import get_user_profile
from smtp_setting import send_email
username = ""
password = ""

hashtag = "flutter"
display_name = "flutter.spirit"
login_url = "https://www.instagram.com/accounts/login/"
search_url = "https://www.instagram.com/explore/tags/"+hashtag+"/"
profile_url = "https://www.instagram.com/"+display_name+"/"

# os.mkdir("./image_data")
# os.mkdir("./image_data/image_0")
# os.mkdir("./image_data/image_1")
# os.mkdir("./image_all")


driver = webdriver.Chrome()
# 인스타그램 접속
driver.get(login_url)
time.sleep(2)
# 로그인
login_result = do_login(driver,username, password)
if login_result == True :
    time.sleep(5)
    driver.get(search_url)
    time.sleep(10)
    # 게시물 총 개수 구하기
    post_list_len = len(driver.find_elements(By.CSS_SELECTOR, ".x9i3mqj"))
    # 첫번째 게시물 클릭
    e = driver.find_element(By.CSS_SELECTOR, ".x9i3mqj")
    e.click()
    for i in range(int(post_list_len) - 10):
        save_post_result = get_post_data(driver)
        if save_post_result == False :
            while save_post_result == True :
                get_post_data(driver)
    data = pd.read_csv('./data/post_data.csv')
    # 사용자 이름 리스트 출력
    profile_list = np.array(data['writer'])
    # 중복 제거
    profile_list = set(profile_list)
    pool = ThreadPool(4)
    save_user_result = pool.map(get_user_profile, profile_list)
    if save_user_result == False :
        while save_user_result == True :
            pool.map(get_user_profile, profile_list)
    else:
        send_email()
elif login_result == False :
    while login_result == True :
        do_login(username, password)


