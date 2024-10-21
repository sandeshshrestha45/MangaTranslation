import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# driver = webdriver.Chrome(ChromeDriverManager().install())
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

driver.maximize_window()

wait = WebDriverWait(driver,20)
def deepl(src_text):

    driver.get(f"https://www.deepl.com/translator#ja/en/{src_text}")


    time.sleep(8)
    text_field_locator= (By.XPATH,'//*[@id="textareasContainer"]/div[3]/section/div[1]/d-textarea/div')
    istext = wait.until(
    lambda driver: driver.find_element(*text_field_locator).get_attribute("value") != "")
    if istext:
        dest = wait.until(EC.presence_of_element_located((By.XPATH,'//*[@id="textareasContainer"]/div[3]/section/div[1]/d-textarea/div')))
        return dest.text
    else :
        return "Could not tranlate"

def google_translate(src_text):
    driver.get(f"https://translate.google.com/?sl=auto&tl=en&text={src_text}&op=translate")

    time.sleep(8)
    text_field_locator =(By.XPATH,'//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[2]/c-wiz[2]/div/div[6]')
    istext = wait.until(
    lambda driver: driver.find_element(*text_field_locator).get_attribute("value") != "")
    if istext:
        dest =wait.until(EC.presence_of_element_located((By.XPATH,'//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[2]/c-wiz[2]/div/div[6]')))
        print(" ".join(dest.text.split("\n")[:-1]))
        processed_text = dest.text.split('\n')
        return " ".join(processed_text[:-1]) if len(processed_text) >1 else processed_text[0]
    else :
        return "Could not translate"

if __name__ == "__main__":

    print(google_translate('全て無効化してしまう！'))