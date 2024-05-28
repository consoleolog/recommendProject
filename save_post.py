from selenium.webdriver.common.by import By
import urllib.request
import time
import uuid
def get_post_data(driver):
    time.sleep(5)
    # 상위 요소 찾기
    try:
        parent_element = driver.find_element(By.CSS_SELECTOR, '._aagv')
        # 이미지 저장
        image = parent_element.find_element(By.TAG_NAME, 'img')
        image = image.get_attribute('src')
        random_name = str(uuid.uuid1()) + '.jpg'
        urllib.request.urlretrieve(image, "./image_all/" + random_name)
        detail_element = driver.find_element(By.CSS_SELECTOR, '.x67bb7w')
        # 좋아요 개수 뽑아내기
        likes = detail_element.find_element(By.CSS_SELECTOR, '.html-span').text
        # 게시물 작성 날자
        post_date = detail_element.find_element(By.CSS_SELECTOR, '.x1p4m5qa').text
        # 해쉬태그 찾기
        content = detail_element.find_elements(By.TAG_NAME, 'a')
        # 작성자 찾기
        writer = detail_element.find_element(By.CSS_SELECTOR, '._acan').text
        file = open("./data/post_data.csv", "a", encoding='utf-8')
        hashtag_list = []
        # 해시태그만 뽑아내기
        for i in range(0, len(content)):
            if content[i].text.find('#') != -1:
                hashtag_list.append(content[i].text)
        file.write(f"\n{str(hashtag_list).replace(',', '/')},{likes},{post_date},{random_name},{writer}")
        file.close()
        next = driver.find_elements(By.CSS_SELECTOR, '.x175jnsf')
        if len(next) == 1:
            next_btn = next[0]
            next_btn.click()
        else:
            next_btn = next[1]
            next_btn.click()
        return True
    except:
        return False

