import requests
import zipfile
import io
import os
import wget
import pandas as pd
import ast
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

FONT_URL = "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"
FONT_FILE = "TaipeiSansTCBeta-Regular.ttf"

def ensure_font(font_url: str = FONT_URL, font_path: str = FONT_FILE):
    if not os.path.isfile(font_path):
        print(f"Downloading font to {font_path}…")
        wget.download(font_url, font_path)
        print(" ✅ Font downloaded.")
    else:
        print("Font already exists, skip download.")

# 下載 zip 檔
def load_and_tfidf():
    # 設定 URL 與儲存路徑
    zip_url = "https://github.com/UrayChenNTHU/udn-game-corner-game-review-article/raw/refs/heads/main/udn-game-corner-game-review-article.zip"
    zip_file_path = "udn-game-corner-game-review-article.zip"
    output_dir = "."
    response = requests.get(zip_url)
    if response.status_code == 200:
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        print("✅ Zip file downloaded successfully.")
    else:
        raise Exception(f"Failed to download file: status {response.status_code}")

    # 解壓縮
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        print("✅ Zip file extracted successfully.")

    data=pd.read_csv(r"./udn game corner game review article dataframe.csv")
    print("✅ CSV loaded successfully.")
    os.remove("udn-game-corner-game-review-article.zip")
    os.remove("udn game corner game review article dataframe.csv")

    data["tokenize and stop words"] = data["tokenize and stop words"].apply(ast.literal_eval)

    data["joined_tokens"] = data["tokenize and stop words"].apply(lambda tokens: " ".join(tokens))

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=2000, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(data["joined_tokens"])
    feature_names = vectorizer.get_feature_names_out()

    def get_tfidf_dict_sparse(idx):
        row = tfidf_matrix.getrow(idx)
        return { feature_names[i]: v for i, v in zip(row.indices, row.data) }
    with ThreadPoolExecutor() as exe:
        tfidf_dicts = list(exe.map(get_tfidf_dict_sparse, range(tfidf_matrix.shape[0])))

    data["tfidf"] = tfidf_dicts
    return data, vectorizer, tfidf_matrix

def draw_wordcloud(data: pd.DataFrame,  vectorizer: TfidfVectorizer ,
                   tfidf_matrix,  query: str = "all", top_n: int = 20):
    print('詞雲製作')
    if query== 'all':
        freq_dict  = {}
        for doc in data["tfidf"]:
            freq_dict .update(doc)
        title = "整體資料庫詞雲"
    else:
        tokens = jieba.lcut(query)
        joined = " ".join(tokens)
        q_vec = vectorizer.transform([joined])

        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        idxs = sims.argsort()[::-1][:top_n]
        freq_dict = {}
        for i in idxs:
            freq_dict.update(data.loc[i, "tfidf"])
        title = f"最相關的前 {top_n} 篇文章詞雲"
    
    print('詞雲產生中...')
    wc = WordCloud(
      width=3000,
      height=3000,
      background_color='white',               #   Background Color
      max_words=100,                    #   Max words
  #    mask=back_image,                       #   Background Image
      #max_font_size=None,                   #   Font size
      font_path="TaipeiSansTCBeta-Regular.ttf",
      random_state=50,                    #   Random color
      regexp=r"\w+(?:[-']\w+)*",  # Update the regexp parameter to include hyphens, you can mark out this line to hide the space character.
      contour_width=1,  # adjust the contour width
      contour_color='black',  # adjust the contour color
      prefer_horizontal=0.9)
    wc.generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    print('詞雲產生完成')
    return fig




if __name__=="__main__":
    data, _ = load_and_tfidf()
    print(data.head())