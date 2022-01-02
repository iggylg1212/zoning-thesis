from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import pickle
import os 
import time
from global_var import ROOT_DIR

class WebScraper(object):
    def __init__(self):
        self.url = 'https://library.municode.com/'
        self.links = []

        self.get_data()

    def get_data(self):
        chrome_options = webdriver.ChromeOptions()
        prefs = {'download.default_directory' : f'{ROOT_DIR}/1data/csv/municode'}
        chrome_options.add_experimental_option('prefs', prefs)
        self.driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', options=chrome_options)

        failed = []

        try: 
            links = pickle.load(open("1data/pickle/state_links.p",'rb'))
        except:
            self.get_links()

        for val_index in range(18,len(links)):
            val = links[val_index]
            cities = list(val.values())[0]
            for index_city in range(len(cities)):
                city = cities[index_city]
                try:
                    self.driver.get(city)
                    # Current 
                    WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '//*[@id="main-menu"]/li[1]/a/div/i')))
                    
                    try: 
                        element = self.driver.find_element_by_xpath('//*[@id="primary"]/div/div[1]/ul/li/div/div[2]/div/p[3]/a')
                        self.driver.execute_script("arguments[0].click();", element)
                    except:
                        pass
                    
                    WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '//*[@id="print-download-toc"]')))
                    element = self.driver.find_element_by_xpath('//*[@id="print-download-toc"]')
                    self.driver.execute_script("arguments[0].click();", element)

                    WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '//*[@id="content"]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[2]/div/div/div/div/div/div/a[2]')))
                    element = self.driver.find_element_by_xpath('//*[@id="content"]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[2]/div/div/div/div/div/div/a[2]')
                    self.driver.execute_script("arguments[0].click();", element)

                    element = self.driver.find_element_by_xpath('/html/body/div[1]/div[2]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[3]/form/button[1]')
                    self.driver.execute_script("arguments[0].click();", element)

                    fileends = "crdownload"
                    while "crdownload" == fileends:
                        time.sleep(1)
                        newest_file = self.latest_download_file()
                        if "crdownload" in newest_file:
                            fileends = "crdownload"
                        else:
                            fileends = "none"

                    element = self.driver.find_element_by_xpath('//*[@id="content"]/ui-view/mcc-codes/div[6]/div/div/div[1]/button/i')
                    self.driver.execute_script("arguments[0].click();", element)

                    element = self.driver.find_element_by_xpath('//*[@id="codebankToggle"]/button')
                    self.driver.execute_script("arguments[0].click();", element)
                    # History 
                    for v in range(2, 25):
                        try: 
                            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.XPATH, f'//*[@id="codebank"]/ul/li[{v}]/div[2]/div/div/button')))
                            element = self.driver.find_element_by_xpath(f'//*[@id="codebank"]/ul/li[{v}]/div[2]/div/div/button')
                            self.driver.execute_script("arguments[0].click();", element)

                            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '//*[@id="codesContent"]/div[2]/mcc-codes-content-landing-page/div/div[3]/div/div[1]/button')))
                            time.sleep(3)
                            element = self.driver.find_element_by_xpath('//*[@id="print-download-toc"]')
                            self.driver.execute_script("arguments[0].click();", element)
                            
                            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '//*[@id="content"]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[2]/div/div/div/div/div/div/a[2]')))
                            element = self.driver.find_element_by_xpath('//*[@id="content"]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[2]/div/div/div/div/div/div/a[2]')
                            self.driver.execute_script("arguments[0].click();", element)

                            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div[2]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[3]/form/button[1]')))
                            element = self.driver.find_element_by_xpath('/html/body/div[1]/div[2]/ui-view/mcc-codes/div[6]/div/div/div[2]/mcc-codes-toc-save-or-print/div[3]/form/button[1]')
                            self.driver.execute_script("arguments[0].click();", element)

                            fileends = "crdownload"
                            while "crdownload" == fileends:
                                time.sleep(1)
                                newest_file = self.latest_download_file()
                                if "crdownload" in newest_file:
                                    fileends = "crdownload"
                                else:
                                    fileends = "none"

                            element = self.driver.find_element_by_xpath('//*[@id="content"]/ui-view/mcc-codes/div[6]/div/div/div[1]/button/i')
                            self.driver.execute_script("arguments[0].click();", element)

                            element = self.driver.find_element_by_xpath('//*[@id="codebankToggle"]/button')
                            self.driver.execute_script("arguments[0].click();", element)
                        
                        except:
                            break
                except:
                    print(val_index, city)
                    failed.append(city)
                    pickle.dump(failed, open(f"{ROOT_DIR}/1data/pickle/failed_links.p", "wb" ))
                    
                    
    def get_links(self):
        options = webdriver.ChromeOptions()
        prefs = {'download.default_directory' : '/1data/csv/municode'}
        chrome_options.add_experimental_option('prefs', prefs)
        self.driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', options=chrome_options)

        self.driver.get(self.url)
        
        lnks = self.driver.find_elements_by_tag_name("a")
        links = []
        for lnk in lnks:
            lnk = lnk.get_attribute('href')
            if (lnk is not None) and ( '#' not in lnk) and ('java' not in lnk):
                strg = (lnk+' ').split('/')[-1]
                if len(strg.strip())==2:
                    links.append(lnk)

        for index in range(len(links)):
            lnk = links[index]
            self.driver.get(lnk)
            state_lnks = self.driver.find_elements_by_tag_name("a")
            state_links = []
            for state_lnk in state_lnks:
                state_lnk = state_lnk.get_attribute('href')
                if (state_lnk is not None) and ( '#' not in state_lnk) and ('java' not in state_lnk):
                    split_list = (state_lnk+' ').split('/')
                    if split_list[-2].strip() == (lnk+' ').split('/')[-1].strip():
                        state_links.append(state_lnk)

            links[index] = {lnk:state_links}
            self.driver.back()
            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div[2]/ui-view/div[2]/section/div/div[1]/div/div/div/map/area[41]')))

        pickle.dump(links, open("1data/pickle/state_links.p", "wb" ))

        self.driver.close()  
    
    def latest_download_file(self):
      os.chdir(f'{ROOT_DIR}/1data/csv/municode')
      files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
      newest = files[-1]

      return newest

if __name__ == '__main__':
    scraper = WebScraper()