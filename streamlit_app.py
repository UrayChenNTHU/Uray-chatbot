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

placeholderstr = "Please input your command"
user_name = "Uray"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

seed = 42

def recommend_games(prompt, df, vectorizer, tfidf_matrix, top_n=5):
    tokens = jieba.lcut(prompt)
    joined = " ".join(tokens)
    vec = vectorizer.transform([joined])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
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
            # 将 few‑shot 示例放到 system_instruction
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

    ctrl_container = st.container()
    with ctrl_container:
        if st.button("🔍請推薦遊戲給我(尚未更新完成)"):
            st.session_state.recommend_triggered = True

        if st.session_state.wc_stage == 0:
            st.button("📋 我想查看詞雲", on_click=to_stage1)

        elif st.session_state.wc_stage == 1:
            st.write("你好，請問你想要看整個資料庫的詞雲，還是你想要尋找特定主題的詞雲？")
            col1, col2 = st.columns(2)
            with col1:
                st.button("📦 我想要全部", on_click=choose_all)
            with col2:
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
