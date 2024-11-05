import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

model = ChatOpenAI(model="gpt-3.5-turbo")

system_summarize_prompt = """
Tóm tắt cần các yếu tố chính của bài báo, nhấn mạnh những thông tin cần thiết, đặc biệt là số liệu quan trọng, trong 2 đến 3 câu.

# Steps
1. Đọc kỹ bài báo để hiểu nội dung chính và các thông tin quan trọng.
2. Xác định các số liệu hoặc sự kiện nổi bật cần được đưa vào tóm tắt để làm nổi bật thông điệp chính của bài báo.
3. Viết đoạn tóm tắt 2 đến 3 câu, tập trung vào các thông tin cốt lõi và số liệu quan trọng.
4. Sử dụng ngôn ngữ sinh động, hấp dẫn để người đọc muốn khám phá tiếp nội dung của bài báo.

# Output Format
- Đoạn văn ngắn 2 đến 3 câu.
- Không bắt đầu bằng cụm từ như “Bài báo nói về” hoặc “Bài báo kể về", thay vào đó sử dụng ngôn ngữ tự nhiên và có sức thu hút.
- Phải thể hiện được thông tin chính của bài báo một cách ngắn gọn và nhấn mạnh vào các số liệu hoặc sự kiện nổi bật.
  
# Notes
Giữ đúng tinh thần và ngôn ngữ của bài báo gốc ({language}), tránh làm mất đi sắc thái nội dung ban đầu.
"""

summarize_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_summarize_prompt),
        ("human", "{content}")
    ]
)

parser = StrOutputParser()

summarize_chain = summarize_prompt_template | model | parser

def clean_content(text: str) -> str:
    """Helper function to clean article content."""
    return text.strip().replace("\n", " ").replace("\t", " ")

def scrape_news(url):
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs[0:-1]])
    html = soup.find('article')
    return clean_content(content), detect(content)

def scrape_news_v2(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text, detect(article.text)

def summarize_news(content, language):
    responses = summarize_chain.stream({
        "content": content,
        "language": language
    })
    for response in responses:
        yield response

if __name__ == "__main__":
    st.title("AI summarize news")
    i = 0
    while i < len(st.session_state.chat_history):
        with st.chat_message("Human"):
            st.markdown(st.session_state.chat_history[i].content)
        with st.chat_message("AI"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Content")
                st.markdown(st.session_state.chat_history[i+1].content)
            with col2:
                st.markdown("### Summary")
                # Assuming the summary is stored in the message.content
                st.markdown(st.session_state.chat_history[i+2].content)

        i += 3


    user_query = st.chat_input("Paste the URL of the news article here:")
    if user_query is not None and user_query != "":
        with st.chat_message("Human"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(user_query))
        with st.spinner("Thinking..."):
            with st.chat_message("AI"):
                content, language = scrape_news_v2(user_query)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Scraped Content")
                    paragraphs = content.split('\n')
                    content = '\n'.join([para for para in paragraphs[0:-1]]).strip()
                    st.write(content)
                st.session_state.chat_history.append(AIMessage(content))
                with col2:
                    st.write("### Summarized Content")
                    answer = st.write_stream(summarize_news(content, language))
                    st.session_state.chat_history.append(AIMessage(answer))