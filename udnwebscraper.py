import re
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import subprocess
import sys
import tempfile

# 確保 BeautifulSoup4 安裝
try:
    from bs4 import BeautifulSoup
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
    from bs4 import BeautifulSoup

# 確保 selenium 安裝
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium"])
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options



__all__ = [
    "fetch_full_page",
    "get_review_list",
    "scrape_udn_game_news_articles",
]

CONTENT_TAG_RE = re.compile(r"'content_tag':\s*\"([^\"]+)\"")
DATE_RE = re.compile(r"'publication_date':\s*'([^']+)'" )


def fetch_full_page(url: str, headless: bool = True, scroll_pause: float = 2.0) -> str:
    """
    使用 Selenium 獲取完整動態載入的頁面 HTML，包括滾動到底部。

    Args:
        url: 目標頁面 URL。
        headless: 是否以 headless 模式啟動瀏覽器。
        scroll_pause: 每次滾動後暫停秒數。

    Returns:
        完整的 HTML 字串。
    """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
       
    driver = webdriver.Chrome(options=chrome_options)
    print('0123456789')
    driver.get(url)

    # 滾動到底部以載入所有內容
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    html = driver.page_source
    driver.quit()
    return html


def get_review_list(category_url: str = 'https://game.udn.com/game/cate/122088', headless: bool = True) -> list[tuple]:
    """
    獲取 UDN-遊戲角落心得評測文章的 URL 與標題列表。

    Args:
        category_url: 分類頁面 URL（預設心得評測分類）。
        headless: 是否使用 headless 模式。

    Returns:
        List of (url, title) tuples。
    """
    html = fetch_full_page(category_url, headless=headless)
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('a', class_='story-list__link')

    reviews = []
    for a in items:
        slot = a.get('data-slotname')
        if slot == 'list_心得評測':
            href = a.get('href')
            title = a.get('title')
            if href and title:
                reviews.append((href, title))
    return reviews


def scrape_single_game_news_article(index: int, url: str, title: str) -> tuple[int, dict]:
    """
    爬取單篇心得評測文章內容。
    Returns a tuple of (index, article_dict).

    article_dict keys:
        url, title, author, author_description,
        publication_date (pd.Timestamp), topics, content
    """
    with requests.Session() as session:
        res = session.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

        # 作者與簡介
        author_el = soup.find('h3', class_='name')
        author = author_el.get_text(strip=True) if author_el else ''
        desc_el = soup.find('div', class_='context-box__text')
        author_desc = desc_el.get_text(' ', strip=True) if desc_el else ''

        # 文章內容段落
        paragraphs = [p for p in soup.find_all('p')]
        content = '\n'.join(p.get_text(' ', strip=True) for p in paragraphs)

        # 取得腳本中的 dataLayer 資訊
        script = soup.find('script', string=re.compile('dataLayer'))
        tag = ''
        pub_date = None
        if script and script.string:
            m = CONTENT_TAG_RE.search(script.string)
            tag = m.group(1) if m else ''
            d = DATE_RE.search(script.string)
            if d:
                pub_date = pd.to_datetime(d.group(1), errors='coerce')

        return index, {
            'url': url,
            'title': title,
            'author': author,
            'author_description': author_desc,
            'publication_date': pub_date,
            'topics': tag,
            'content': content
        }


def scrape_udn_game_news_articles(url_list: list[tuple], max_workers: int = 10) -> pd.DataFrame:
    """
    多執行緒爬取多篇文章，並以 DataFrame 回傳，保持原始順序。

    Args:
        url_list: list of (url, title)
        max_workers: 最大平行執行緒數量

    Returns:
        pandas.DataFrame: columns=['url','title','author',...]
    """
    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (url, title) in enumerate(url_list):
            futures.append(executor.submit(scrape_single_game_news_article, idx, url, title))

        for future in tqdm(as_completed(futures), total=len(futures), desc='爬取進度'):
            idx, art = future.result()
            results.append((idx, art))

    # 按照 index 排序，再轉成 DataFrame
    results.sort(key=lambda x: x[0])
    articles = [a for _, a in results]
    return pd.DataFrame(articles)


if __name__ == '__main__':
    # 簡易測試
    urls = get_review_list()
    df = scrape_udn_game_news_articles(urls)
    print(df.head())
