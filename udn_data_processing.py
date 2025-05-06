import requests
import zipfile
import io
import os
import pandas as pd
import ast
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer



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



if __name__=="__main__":
    data, _ = load_and_tfidf()
    print(data.head())