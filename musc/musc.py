import os
import time
import requests
import string
import pandas as pd
import shutil
import traceback
import gender_guesser.detector as gender
from ethnicolr import pred_fl_reg_name, pred_census_ln
from deepface import DeepFace
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tenacity import retry, stop_after_attempt, wait_exponential

DATA_DIR = os.path.join(os.path.expanduser('~'), 'data/musc_directory')
os.makedirs(DATA_DIR, exist_ok=True)

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

def extract_info(row):
    def find_text(tag, attr=None, value=None, id_endswith=None):
        try:
            if id_endswith: return row.find(id=lambda x: x and x.endswith(id_endswith)).text.strip()
            return row.find(tag, **{attr: value}).text.strip() if attr else row.find(tag).text.strip()
        except: return None

    def find_image_src(tag, attr=None, value=None):
        try:
            return row.find(tag, **{attr: value})["src"]
        except: return None

    return {key: (find_image_src(*args) if key == "image" else find_text(*args)) for key, args in {
        "name": ("a", "class_", "ProviderName"),
        "image": ("input", "class_", "phy-photo"),
        "college": ("div", None, None, "CollegeLabel"),
        "department": ("div", None, None, "DeptLabel"),
        "rank": ("div", None, None, "TitleLabel"),
        "phone": ("div", None, None, "PhoneLabel"),
        "email": ("div", None, None, "EmailLabel"),
    }.items()}

def process_page(driver):
    return [extract_info(row) for row in BeautifulSoup(driver.page_source, "html.parser").find_all("tr", class_=["rgRow", "rgAltRow"])]

for letter in string.ascii_uppercase:
    url = f"https://education.musc.edu/MUSCApps/facultydirectory/FacultyAlpha.aspx?List={letter}"
    try:
        driver.get(url)
        data = []
        while True:
            records = process_page(driver)
            if all([r['name'] in {e['name'] for e in data} for r in records]): break
            data.extend(records)
            time.sleep(1)
            if not (next_buttons := driver.find_elements(By.CLASS_NAME, "rgPageNext")): break
            next_buttons[0].click()
            time.sleep(1)
        pd.DataFrame(data).to_json(os.path.join(DATA_DIR, f"json/PAGE_{letter}.json"), orient='records', lines=True)
    except: traceback.print_exc()

driver.quit()

GENDER_DETECTOR = gender.Detector()

def predict_gender(first_name): return GENDER_DETECTOR.get_gender(first_name)

def predict_ethnicity(first_name, last_name):
    df = pd.DataFrame([{'first': first_name, 'last': last_name} if first_name else {'last': last_name}])
    predictions = pred_fl_reg_name(df, 'last', 'first') if first_name else pred_census_ln(df, 'last', 'first')
    return predictions['race'].iloc[0]

def predict_image_demographics(image_path):
    try:
        demographics = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race'], silent=True)
        if len(demographics) == 0:
            return (None,)*3
        demographics = demographics[0]
        return demographics['age'], demographics['dominant_gender'], demographics['dominant_race']
    except: traceback.print_exc(); return (None,)*3

def predict_demographics(name, image_path=None):
    last_name, first_name, *_ = map(lambda x: x.replace(',', ''), name.split())
    name_gender, name_ethnicity = predict_gender(first_name), predict_ethnicity(first_name, last_name)
    image_age, image_gender, image_ethnicity = predict_image_demographics(image_path) if image_path else (None,) * 3
    return dict(
        name_gender=name_gender,
        name_ethnicity=name_ethnicity,
        image_age=image_age,
        image_gender=image_gender,
        image_ethnicity=image_ethnicity
    )

def load_data():
    data = []
    for letter in string.ascii_uppercase:
        data.extend(pd.read_json(os.path.join(DATA_DIR, f"json/PAGE_{letter}.json"), orient='records', lines=True)
                    .assign(directory_page=letter).to_dict(orient='records'))
    return data

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def get_with_retry(url): return requests.get(url, stream=True)

records, results = load_data(), []

for record in records:
    name, image_url = record['name'], record['image']
    image_path = None
    if image_url is not None:
        filename = ''.join(c for c in os.path.basename(image_url) if c in "-_.%s%s" % (string.ascii_letters, string.digits))
        image_path = os.path.join(DATA_DIR, f"images/{filename}")
        if not os.path.exists(image_path):
            with open(image_path, 'wb') as f:
                shutil.copyfileobj(get_with_retry(image_url).raw, f)

    results.append({
        **record,
        **dict(image_path=image_path),
        **predict_demographics(name, None if 'NoImageProvided.png' in image_path else image_path)
    })

employees = pd.DataFrame(results)
employees.to_parquet(os.path.join(DATA_DIR, f"data.parquet"))
