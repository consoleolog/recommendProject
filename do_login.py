from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
def do_login(driver,username, password):
    try:
        # 아이디 인풋 선택
        e = driver.find_element(By.CSS_SELECTOR, 'input[name="username"]')
        e.click()
        e.send_keys(username)
        # 비번 인풋 선택
        e = driver.find_element(By.CSS_SELECTOR, 'input[name="password"]')
        e.click()
        e.send_keys(password)
        e.send_keys(Keys.ENTER)
        return True
    # 로그인 실패 하면 다시 처음으로
    except:
        return False