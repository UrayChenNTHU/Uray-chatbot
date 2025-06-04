from openai import OpenAI
import time
import re
from dotenv import load_dotenv
import os
import subprocess
import sys
import udn_data_processing
from sklearn.metrics.pairwise import cosine_similarity
import json
from google import genai
from google.genai import types
import jieba
import ast
import pandas as pd
import logging
import io
from PIL import Image

# Import ConversableAgent class
import autogen
from autogen import ConversableAgent, LLMConfig
from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from autogen.code_utils import content_str
from coding.constant import JOB_DEFINITION, RESPONSE_FORMAT

import streamlit as st


@st.cache_data(show_spinner=False)
def load_data():
    return udn_data_processing.load_and_tfidf()


# Load environment variables from .env file
load_dotenv(override=True)

# https://ai.google.dev/gemini-api/docs/pricing
# URL configurations
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPEN_API_KEY = st.secrets["OPEN_API_KEY"]
DEEPSEEK_API_KEY = st.secrets["Deepseek_KEY"]

placeholderstr = "Please input your command"
user_name = "User"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

seed = 42

'''
llm_config_tokenizer = LLMConfig(
    api_type="openai",
    model="gpt-4o-mini",
    api_key=OPEN_API_KEY,
)

with llm_config_tokenizer:
    tokenizer_agent = ConversableAgent(
        name="tokenizer_agent",
        system_message=(
            "你是一個中文斷詞專家。\n"
            "輸入：一段中文文字。請對它做斷詞，將所有詞語以 Python list 的形式回傳。\n"
            "例如：輸入「我想玩動作遊戲」，應回傳 ['我', '想', '玩', '動作', '遊戲']。\n"
            "請務必只輸出一個合法的 Python list，不要額外加任何解釋文字。"
        )
    )

def tokenize_by_llm(text: str) -> list:
    """
    把 text 交給 LLM，要求它直接回傳 Python list 形式的 token list。
    若失敗就回傳空列表或原始文字做 fallback。
    """
    prompt = f"請對以下文字做中文斷詞，並以 Python list 回傳：\n\n{text}"
    try:
        result = tokenizer_agent.run(task=prompt)
        return ast.literal_eval(result.final_output.strip())
    except Exception as e:
        logging.error(f"LLM 斷詞失敗：{e}")
        return []

def recommend_games(prompt, df, vectorizer, tfidf_matrix, top_n=5):
    # 1. 用 LLM 做斷詞
    tokens = tokenize_by_llm(prompt)
    # 2. 如果 LLM 回傳空 list，fallback 把整段當一個 token
    if not isinstance(tokens, list) or len(tokens) == 0:
        tokens = [prompt]
    # 3. 用空格串起 tokens，再做向量化
    joined = " ".join(tokens)
    vec = vectorizer.transform([joined])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1][:top_n]
    recs = df.iloc[idxs].copy()
    recs["score"] = sims[idxs]
    return recs
'''

def recommend_games(prompt: str, df: pd.DataFrame, vectorizer, tfidf_matrix, top_n: int = 5):
    """
    總入口：先請 LLM 拆解「喜歡」跟「不喜歡」，
    再把拆出來的片段各自做 TF-IDF，最後合併向量算相似度。
    """
    # 1. 用 LLM 拆成 like/dislike
    prefs = split_preferences_by_llm(prompt)
    print(prefs)
    positive_text = prefs.get("like", "").strip()
    negative_text = prefs.get("dislike", "").strip()

    # 2. 定義把文字轉成向量的 helper
    def text_to_vec(text: str):
        tokens = jieba.lcut(text)
        joined = " ".join(tokens)
        return vectorizer.transform([joined])

    # 3. 如果 positive 也空，就直接 fallback 成原本舊版（全部當成一段正面）
    if not positive_text:
        positive_text = prompt

    pos_vec = text_to_vec(positive_text)
    neg_vec = None
    if negative_text:
        neg_vec = text_to_vec(negative_text)

    # 4. 合併向量：pos - neg
    if neg_vec is not None:
        combined_vec = pos_vec - neg_vec
    else:
        combined_vec = pos_vec

    # 5. 用合成向量跟整個 tfidf_matrix 算 cosine similarity
    sims = cosine_similarity(combined_vec, tfidf_matrix).flatten()

    # 6. 挑 top_n
    idxs = sims.argsort()[::-1][:top_n]
    recs = df.iloc[idxs].copy()
    recs["score"] = sims[idxs]

    return recs

llm_config_gemini = LLMConfig(
    api_type = "google", 
    model="gemini-2.0-flash-lite",                    # The specific model
    api_key=GEMINI_API_KEY,   # Authentication
)

llm_config_openai = LLMConfig(
    api_type = "openai", 
    model="gpt-4o-mini",                    # The specific model
    api_key=OPEN_API_KEY,   # Authentication
)

with llm_config_gemini:
    assistant = AssistantAgent(
        name="assistant",
        system_message=(
        "You are a helpful assistant. "
        "Answer user questions appropriately. "
        "After your result, say 'ALL DONE'. "
        "Do not say 'ALL DONE' in the same response."
        ),
        max_consecutive_auto_reply=2
    )

with llm_config_openai:
    tokens_refiner = ConversableAgent(
        name="tokens_refiner",
        system_message=(
            "You are a Chinese token refinement expert.\n"
            "Input: a list of tokens, already stop-word filtered.\n"
            "Remove control characters, punctuation, meaningless tokens, and merge game-specific terminologies.\n"
            "Return the cleaned tokens as a Python list."
        )
    )


# 3. Helper to refine tokens via LLM
def refine_by_llm(token_list):
    prompt = f"Refine the following token list: {token_list}"
    try:
        
        result = tokens_refiner.run(task=prompt)
        
        return ast.literal_eval(result.final_output)
    except Exception as e:
        logging.error(f"Error refining tokens: {e}")
        return token_list

def refine_tokens_list(token_list):
    if not isinstance(token_list, list):
        return token_list
    
    return refine_by_llm(token_list)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
def split_preferences_by_llm(user_input: str) -> dict:
    """
    用 Google Gemini 把一句含「喜歡」和「不喜歡」的中文拆成 JSON：
      { "like": "...", "dislike": "..." }
    如果 Gemini 回傳非 JSON，就 fallback 只給 like，把 dislike 設空。
    """
    system_instruction = (
        "你是一個中文語意分析專家，"
        "能把一段包含「喜歡」跟「不喜歡」的句子拆成兩個欄位："
        "'like' 跟 'dislike'，以 JSON 回傳。"
        '例如原句為「我喜歡可樂，但是不喜歡牛奶」則JSON為：{ "like": "可樂", "dislike": "牛奶" }' \
        '「我不喜歡上學，也不喜歡玩耍」則JSON為：{ "like": "...", "dislike": "上學、玩耍" }' \
        '「我喜歡看海，也喜歡爬山」則JSON為：{ "like": "看海、爬山", "dislike": "..." }'
        "請僅回傳純 JSON，千萬不要多加任何解釋文字。\n"
        "如果句子裡沒有「不喜歡」的部分，就把 dislike 設為空字串。"
        "中文說明請使用繁體中文。"
    )
    user_message = f"請把這段拆成 JSON：\"{user_input}\""

    # 2. 呼叫 Gemini
    resp = gemini_client.models.generate_content(
        model="gemini-2.0-flash-lite",  # 或你有權限的其他 Gemini 型號
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=200
        )
    )

    text = resp.text.strip()
    try:
        # 嘗試 parse 回傳結果裡的 JSON
        obj = json.loads(text)
        like = obj.get("like", "").strip()
        dislike = obj.get("dislike", "").strip()
        return {"like": like, "dislike": dislike}
    except Exception:
        # parse 失敗就 fallback
        return {"like": user_input, "dislike": ""}


user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0,
)

# Function Declaration 

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.05)

def save_lang():
    st.session_state['lang_setting'] = st.session_state.get("language_select")

def paging():
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/two_agents.py", label="Two Agents' Talk", icon="💭")

def append_wordcloud_to_history(df, vectorizer, tfidf_matrix, query):
    fig = udn_data_processing.draw_wordcloud(df, vectorizer, tfidf_matrix, query=query)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = buf.read()
    st.session_state.messages.append({
    "role": "assistant",
    "type": "wordcloud",
    "image_bytes": img_bytes,
    "text_before": f"好的，正在為你生成「{query}」的詞雲…",
    "text_after": "這是你的詞雲，希望對你有幫助！"
})

def main():
    udn_data_processing.ensure_font()

    if "wc_stage" not in st.session_state:
        st.session_state.wc_stage = 0
    if "wc_query" not in st.session_state:
        st.session_state.wc_query = ""
    if "recommend_triggered" not in st.session_state:
        st.session_state.recommend_triggered = False
    if "recommend_query" not in st.session_state:
        st.session_state.recommend_query = ""

    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    df, vectorizer, tfidf_matrix = load_data()
    # Show title and description.
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        paging()
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=0, on_change=save_lang, key="language_select")
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting
        

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")

    st_c_chat = st.container(border=True)
    with st_c_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        else:
            for msg in st.session_state.messages:
                if msg.get("type") == "wordcloud":
                    st.chat_message("assistant", avatar=user_image).write(msg["text_before"])
                    st.chat_message("assistant", avatar=user_image).image(Image.open(io.BytesIO(msg["image_bytes"])))
                    st.chat_message("assistant", avatar=user_image).write(msg["text_after"])
                elif msg["role"] == "user":
                    if user_image:
                        st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                    else:
                        st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
                elif msg["role"] == "assistant":
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
                else:
                    try:
                        image_tmp = msg.get("image")
                        if image_tmp:
                            st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                    except:
                        st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))


        story_template = ("Please give appropriate response according to the content of ##PROMPT##."
                        f"And remeber to mention user's name {user_name} in the end. Don't add some emoji in the end of each sentence."
                        f"Please express in {lang_setting}")

        classification_template = ("You are a classification agent, your job is to classify what ##PROMPT## is according to the job definition list in <JOB_DEFINITION>"
        "<JOB_DEFINITION>"
        f"{JOB_DEFINITION}"
        "</JOB_DEFINITION>"
        "Please output in JSON-format only."
        "JSON-format is as below:"
        f"{RESPONSE_FORMAT}"
        "Let's think step by step."
        f"Please output in {lang_setting}"
        )

        def generate_response(prompt):

            # prompt_template = f"Give me a story started from '{prompt}'"
            prompt_template = story_template.replace('##PROMPT##',prompt)
            # prompt_template = classification_template.replace('##PROMPT##',prompt)
            result = user_proxy.initiate_chat(
            recipient=assistant,
            message=prompt_template
            )

            response = result.summary
            return response

        def show_chat_history(chat_hsitory):
            for entry in chat_hsitory:
                role = entry.get('role')
                name = entry.get('name')
                content = entry.get('content')
                st.session_state.messages.append({"role": f"{role}", "content": content})

                if len(content.strip()) != 0: 
                    if 'ALL DONE' in content:
                        return 
                    else: 
                        if role != 'assistant':
                            st_c_chat.chat_message(f"{role}").write((content))
                        else:
                            st_c_chat.chat_message("user",avatar=user_image).write(content)
        
            return 
        
        def wants_recommendation(prompt: str) -> bool:
            sys_instruction = """
        You are an intent classifier. Decide whether the user is asking for game recommendations.
        As long as there are words related to the game and recommend(遊戲、推薦), it can be regarded as the user wants to be recommended
        Respond in JSON ONLY, with a boolean field "recommend".
        Some Examples:
        Input: "我想找一些好玩的遊戲"
        Output: {"recommend": true}
        Input: "今天天氣怎麼樣？"
        Output: {"recommend": false}
        Input: "幫我推薦 RPG 類型的遊戲"
        Output: {"recommend": true}
        Input: "你喜歡什麼顏色？"
        Output: {"recommend": false}
        Now classify the final input.  (Do NOT output anything else.)
        """
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    max_output_tokens=50
                )
            )

            text = resp.text.strip()
            match = re.search(r'\{.*\}', text)
            if match:
                try:
                    j = json.loads(match.group())
                    return bool(j.get("recommend", False))
                except json.JSONDecodeError:
                    pass
            return False
        
        def chat(prompt: str):
            st_c_chat.chat_message("user",avatar=user_image).write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            if wants_recommendation(prompt):
                recs = recommend_games(prompt, df, vectorizer, tfidf_matrix, top_n=5)
                st.markdown("### 🎯 我猜你可能有興趣的文章／遊戲")
                st.table(
                    recs[["title","url","score"]]
                    .assign(score=lambda df: df["score"].map(lambda x: f"{x:.3f}"))
                )
                context = ""
                for _, row in recs.iterrows():
                    snippet = row["content"][:300].replace("\n"," ")
                    context += f"文章標題：{row['title']}\n摘要：{snippet} …\n\n"

                # 4. 讓 LLM 根據這些內容做簡短介紹
                summary_prompt = (
                    "以下是兩篇遊戲心得文章的標題與內容摘要，"
                    "請分別用 2‑3 句話，介紹這兩款遊戲的主要特色與玩法：\n\n"
                    f"{context}"
                )
                intro = generate_response(summary_prompt)

                st.markdown("### 📖 推薦遊戲簡介")
                st.write(intro)
            else:
                reply = generate_response(prompt)
                st_c_chat.chat_message("assistant").write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            
    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

    def to_stage1():
        st.session_state.wc_stage = 1

    def choose_all():
        append_wordcloud_to_history(df, vectorizer, tfidf_matrix, query="all")
        st.session_state.wc_stage = 0

    def to_stage2():
        st.session_state.wc_stage = 2

    def submit_query():
        q = st.session_state.wc_query.strip()
        if q:
            append_wordcloud_to_history(df, vectorizer, tfidf_matrix, query=q)
        st.session_state.wc_query = ""
        st.session_state.wc_stage = 0

    def trigger_recommend():
            st.session_state.recommend_triggered = True
    
    def reset_recommend():
        st.session_state.recommend_triggered = False
        st.session_state.recommend_query = ""

    def run_recommend_flow(user_input):
            """
            這段負責呼叫既有的 recommend_games() + generate_response()，顯示結果。
            """
            st.session_state.recommend_triggered = False
            st.session_state.pop("recommend_query", None)
            recs = recommend_games(user_input, df, vectorizer, tfidf_matrix, top_n=5)
            df_for_md = recs[["title", "url", "score"]].copy()
            df_for_md["score"] = df_for_md["score"].map(lambda x: f"{x:.3f}")
            df_for_md["title"] = df_for_md.apply(lambda row: f"[{row['title']}]({row['url']})", axis=1)
            md_table = df_for_md.to_markdown(index=False)
            context = ""
            for _, row in recs.iterrows():
                snippet = row["content"][:300].replace("\n", " ")
                context += f"文章標題：{row['title']}\n摘要：{snippet} …\n\n"
            summary_prompt = (
                "以下是兩篇遊戲心得文章的標題與內容摘要，"
                "請分別用 2-3 句話，介紹這兩款遊戲的主要特色與玩法：\n\n"
                f"{context}"
            )
            intro = generate_response(summary_prompt)
            markdown_str = (
                "### 🎯 我猜你可能有興趣的文章／遊戲\n\n"
                f"{md_table}\n\n" 
                "### 📖 推薦遊戲簡介\n\n"
                f"{intro}"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": markdown_str
            })

    def on_submit_recommend():
        user_qry = st.session_state.get("recommend_query", "").strip()
        if user_qry:
            run_recommend_flow(user_qry)
        else:
            st.warning("請先輸入想要搜尋的關鍵字／描述")


    ctrl_container = st.container()
    with ctrl_container:
        if not st.session_state.recommend_triggered:
            st.button("🔍請推薦遊戲給我", on_click=trigger_recommend)
        else:
                st.markdown(
                    "沒問題！請輸入你想知道的主題元素，可以是字詞也可以是一段描述，"
                    "如果描述得越清楚，我越能更準確地推薦你想要的遊戲哦！"
                )
                user_qry = st.text_input("", key="recommend_query", placeholder="請在此輸入推薦關鍵字")
                st.button("提交", on_click=on_submit_recommend)
            

        if st.session_state.wc_stage == 0:
            st.button("📋 我想查看詞雲", on_click=to_stage1)

        elif st.session_state.wc_stage == 1:
            st.write("你好，請問你想要看整個資料庫的詞雲，還是你想要尋找特定主題的詞雲？")
            st.button("📦 我想要全部", on_click=choose_all)
            st.button("🔍 我想知道特定主題的詞雲", on_click=to_stage2)
        
        if st.session_state.wc_stage == 2:
            st.markdown("沒問題！請輸入你想知道的主題元素，可以是字詞也可以是一段描述，如果描述得越清楚，我越能更準確的找到你想要的詞雲哦！")
            query = st.text_input(
            "", key="wc_query")
            if st.button("提交", on_click=submit_query):
                pass
       
        with st.expander("📊 顯示原始資料與 LLM 處理結果"):
            st.subheader("Preview of Raw Data")
            st.dataframe(df)

            raw_col = 'tokenize and stop words without remove control'
            if raw_col in df.columns:
                st.subheader("Refining tokens with LLM skill")
                df['tokens_refined'] = df[raw_col].apply(refine_tokens_list)
                st.dataframe(df[[raw_col, 'tokens_refined']])
            else:
                st.warning(f"Column '{raw_col}' not found in CSV.")

if __name__ == "__main__":
    main()
