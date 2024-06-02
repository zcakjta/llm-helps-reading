import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
#from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser

def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

def count_words(context):
    # Split text into segments of Chinese and non-Chinese
    segments = re.split(r'([\u4e00-\u9FFF]+)', context)
    
    word_count = 0
    for segment in segments:
        if re.search('[\u4e00-\u9FFF]', segment):
            # For Chinese segments, count continuous characters as words
            word_count += len(re.findall(r'[\u4e00-\u9FFF]', segment))
        else:
            # For non-Chinese segments, count words using space as a delimiter
            word_count += len(re.findall(r'\b\w+\b', segment))
    
    return word_count    


def calculate_reading_time(word_count, words_per_minute=300):
    # Calculate the reading time in minutes and round to the nearest whole number
    reading_time = round(word_count / words_per_minute)
    if reading_time < 1:
         reading_time = 1
    return reading_time
def generate_response(content,temperature,option):
    is_url = content.startswith('http')
    llm.temperature = temperature
    if option == "快速总结":
        response = quick_chain.invoke({"content": content,"is_url":is_url})
    else:
        response = pro_chain.invoke({"content": content,"is_url":is_url})
    summary = response.get('summary', 'No summary available')
    word_count = response.get('word_count', 'No word count available') if is_url else count_words(content)
    return content, summary, word_count

summary_template_pro =summary_template_pro = """ 
Provide a comprehensive summary of the content in context:
Your main goal here is to help me understand the key ideas of the article. 
This consists of two parts: first I can understand what the whole article is about and this is what you will do in paragraph 1; 
Second I can understand the key points that are illustrated with details so that i do need to refer back to the whole article. Your task should resemble the work from the process of a human reads the article and absorbs the knowledge
and produce the notes\n

Your summarization should consist of 2 paragraphs.

Paragraph 1: You should first describe the whole article in one paragraph no shorter than 300 characters. This helps me skim through the information \n
Paragraph 2: you should provide summaries of the main points for each section by first reading through the whole article, and then breaking the information down into sections, and identifying the key points in each section
To provide summaries of the key points for each section, you should follow the below strcuture and requirements as below \n
1.主题: this is for you to use one sentence to describe the section. \n
2.观察与发现: you need to extract the main findings talked about in the section. The main findings should be illustrated with details to help me better understand \n
3.观点: you need to paraphrase the main argument brought in this section. The main argument should be very informative with details; \n
4.论据: you need to list the all evidences that supports the main argument. the evidences need to be illustrated with details to help me better understand
\n
for Paragraph 2, you will have to explain the main points comprehensively with maximum depth of informative details so that I can read it and understand 
it without needing to referencing back to the whole article to understand the nuances. Remember, it is not just a list of events or a vague overview
you should form a summrization that is well strcutured and informative and in depth, with facts and numbers if available and a minimum of 1,200 words.
Aim for a summary length of approximately 40 percent of the original text
You can also directly reference the original content in the context if it is well written
You should strive to write the summrization as long as you can using all relevant and necessary information provided.
You should maintain the logical flow, ensuring the summary is conherent and easy to follow
If multiple sections in the context all mention very similar topics, merge them together and do not produce repeated content across different sections\n
You must write the summerization with markdown syntax.
If context is not available or has no practical meaning (as lots of scraping prevention program will return some nonsense), please respond as "no content available"\n
Your response must be provided in Chinese.  \n
Please do your best, this is very important to my career\n
Context:
{context}
"""
summary_prompt_pro = PromptTemplate.from_template(summary_template_pro)

summary_template_quick = """ 

Summerize the article provided in the context:

Your main goal here is to provide a narrative summary for the users to understand what the context is about.
This helps the users to decide whether they want to read the context in details
Below is the requirement for the narrative summary \n
[“A narrative summary is a way to condense an article into a shorter form.”

Narrative summaries focus on the core events, maintaining a sense of the original narrative flow.\n
Key Features:
Concise: Focuses on essential plot points and character actions.
Chronological: Maintains the flow of events from beginning to end.
Tells, more than shows: Often uses direct statements rather than extensive dialogue or scene descriptions.]\n
Your response must be provided in Chinese. You must write at least 100 words for the narrative summary.

{context}
"""
summary_prompt_quick = PromptTemplate.from_template(summary_template_quick)

further_questions ="""
You are provided with the full context and have generated a summary for the user to have a general understanding
of the context.\n
full context:{context} \n
summary: {summary} \n
However, users might have further questions to clarify, Please come up with 3 follow-up questions that users want to further ask about\n
the summary and the full context.
User may have particularly interest: that is enclosed in [] below,, please adjust the 3 follow-up questions if user's interest provided and relevant\n
User's interest:[{user_input}]
You must respond with a list of 3 strings in the following format,:
"[\"question 1\", \"question 2\", \"question 3\"]". don't forget about the slashes, otherwise it will cause errors
You must provide the response in Chinese
If provided content suggests no practical meaning or non-sense as it is often the case with scraping protection program, return no further questions available
"""

further_questions_prompt = PromptTemplate.from_template(further_questions)


further_question_answer ="""
You are provided with the full context and have generated a summary for the user to have a general understanding
of the context.\n
full context:{context} \n
summary: {summary} \n
Please use the information to answer user's question : {question} in depth with informative details from the context.
You must provide the response in Chinese.
You must write the response with markdown syntax.
please do your best. This is very important to my career
"""
further_questions_answer_prompt = PromptTemplate.from_template(further_question_answer)






### Streamlit APP
import streamlit as st
import streamlit_card

st.set_page_config(page_title="🤓📖 GPT阅读总结助手_V1.0", page_icon='📖',layout='wide')
st.title('🤓📖 GPT阅读总结助手_V0.1')
st.caption('💬 让阅读更加高效\nPowered By Streamlit & OpenAI')

def update_session_state(word_count,content,summary,markdown_content):
  st.session_state['summary'] = summary
  st.session_state['content'] = content
  st.session_state['word_count'] = word_count
  st.session_state['markdown_content'] =markdown_content

import time
def stream_data(stream_content):
    for text in stream_content:
        yield text
        time.sleep(0.02)


from streamlit_card import card


col1_card,col2_card,col3_card =st.columns([1,1,1])

with col1_card:
    csdn_card = card(
    title="",
    text="AI点评 from AI_GUMP",
    image="https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/IBM_Electronic_Data_Processing_Machine_-_GPN-2000-001881.jpg/1200px-IBM_Electronic_Data_Processing_Machine_-_GPN-2000-001881.jpg",
    url="https://blog.csdn.net/AI_Gump",
        styles={
        "card": {
            "width": "260px", 
            "height": "260px" 
        }
    }
)

with col2_card:
    reddit_card = card(
    title="",
    text="AI资讯 from LocalLLaMA",
    image="https://www.naesp.org/wp-content/uploads/2023/08/Llama-scaled.jpeg",
    url="https://www.reddit.com/r/LocalLLaMA/",
        styles={
        "card": {
            "width": "260px", 
            "height": "260px" 
        }
    }
)

with col3_card:
    twitter_card = card(
    title="",
    text="AI前沿观点 from Yann Lecun",
    image="https://pplx-res.cloudinary.com/image/fetch/s--qV2dgJFG--/t_limit/https://engineering.fb.com/wp-content/uploads/2019/03/Yann.jpg",
    url="https://x.com/ylecun",
        styles={
        "card": {
            "width": "260px", 
            "height": "260px" 
        }
    }

)

openai_api_key = st.sidebar.text_input('OpenAI API Key',key = 'chatbot_api_key',type='password')

temperature = st.sidebar.slider(
        '模型回复随机性',
        min_value=0.0,
        max_value=1.0,
        value =0.3,
        step = 0.1
    )
option = st.sidebar.radio("总结类型:",("快速总结","深度总结"))


llm = ChatOpenAI(model="gpt-4o-2024-05-13",temperature=0.0,streaming=True,api_key=openai_api_key)
llm_quick = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0,streaming=True,api_key=openai_api_key)
pro_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
        context = lambda x: scrape_text(x["content"]) if x["is_url"] else x["content"]
    ) | summary_prompt_pro |llm| StrOutputParser(),word_count = lambda x: count_words(scrape_text(x["content"]))if x["is_url"] else x["content"]
)

quick_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
        context = lambda x: scrape_text(x["content"]) if x["is_url"] else x["content"]
    ) | summary_prompt_quick |llm| StrOutputParser(),word_count = lambda x: count_words(scrape_text(x["content"]))if x["is_url"] else x["content"]
)

further_questions_chain = further_questions_prompt|llm|StrOutputParser()|json.loads
further_questions_answer_chain = further_questions_answer_prompt|llm|StrOutputParser()


with st.form('my_form'):
        input_text = st.text_input(label= '💡 今天想读点什么呢?'
            ,placeholder='✍🏼 请输入想要总结的文章链接或者手动复制粘贴内容')
        submitted = st.form_submit_button('提交')
if st.button("重置"):
    # Clear all items from st.session_state
    st.session_state.clear()
    st.experimental_rerun()

if 'markdwon_content' in st.session_state:
        st.markdown(st.session_state['markdown_content'])

if submitted:
        if not openai_api_key:
             st.info("请在侧边栏输入API key")
             st.stop()
        if st.session_state:
            st.session_state.clear()
            st.session_state['chatbot_api_key'] = openai_api_key
        with st.spinner('✍🏼阅读助手总结中....'):
            content, summary, word_count = generate_response(input_text,temperature,option)           
            reading_time = calculate_reading_time(word_count)
            
            is_url = input_text.startswith('http')
            if is_url:       
                markdown_content = f"📖 原文长度 {word_count}字, ⏩ 全文阅读耗时约 {reading_time}分钟, [点击这里阅读全文]({content})\n\n## 阅读笔记\n\n{summary}"
            else:
                markdown_content = f"📖 原文长度 {word_count}字, ⏩ 全文阅读耗时约 {reading_time}分钟\n\n## 阅读笔记\n\n{summary}"

            update_session_state(word_count,content,summary,markdown_content)

            st.write_stream(stream_data(markdown_content))

 
def button_clicked(button_name):
    content = st.session_state.get('content', '')
    summary = st.session_state['summary']
    word_count = st.session_state['word_count']
    markdown_content = st.session_state['markdown_content']
    with st.spinner(f"🤔阅读助手思考中: {button_name}"):
        context = scrape_text(content) if content.startswith('http') else content 
        retrieval_answer = further_questions_answer_chain.invoke({"context":context ,"summary":summary,"question":button_name})
        st.session_state['retrieval_answer'] = retrieval_answer
        st.session_state['button_clicked']=button_name
        st.session_state['messages'].append({'role':'user','avatar':'👩🏼‍💻','text':button_name})

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","avatar":"🤖","text":"👋 嗨，有什么想要进一步了解的吗？我可以帮忙！-2 "}]
   
for msg in st.session_state.messages[1:]:
    st.chat_message(msg['role'],avatar=msg['avatar']).write(msg['text']) 


if 'markdown_content' in st.session_state:             
    if prompt:= st.chat_input("💭 想进一步了解哪些文章详细信息？"):


                
        with st.spinner(f'🤔阅读助手思考中:{prompt}'):
  
            content = st.session_state.get('content', '')
            summary = st.session_state['summary']
            context = scrape_text(content) if content.startswith('http') else content

            retrieval_answer = further_questions_answer_chain.invoke({"context": context, "summary": summary, "question": prompt})

            st.session_state['messages'].append({'role':'user','avatar':'👩🏼‍💻','text':prompt})
            st.session_state['retrieval_answer'] = retrieval_answer 

            st.write('individual -1 ')
            st.chat_message('user',avatar='👩🏼‍💻').write(prompt)
            st.write('individual -1 -end')
                    
    st.info('系统提示：more questions to come')
 
    summary = st.session_state['summary']
    content = st.session_state['content']

    if 'further_questions_lists' not in st.session_state:
        further_questions_lists = further_questions_chain.invoke({"context":scrape_text(content) if input_text.startswith('http') else input_text,"summary":summary,"user_input":''})
        st.session_state['further_questions_lists'] = further_questions_lists
    else:
        for msg in reversed(st.session_state['messages']):
            further_questions_lists = further_questions_chain.invoke({"context":scrape_text(content) if input_text.startswith('http') else input_text,
                                                        "summary":summary,
                                                        "user_input":msg['text'] if msg['role'] =='user' else None}) 
            st.write('generate new questions') 
            st.session_state['further_questions_lists'] = further_questions_lists
            break
 
    with st.chat_message("assistant",avatar='🤖'):
        st.write("👋 嗨，有什么想要进一步了解的吗？我可以帮忙！-1 ")
        with st.expander('📖阅读笔记'):
            st.markdown(st.session_state['markdown_content'])      
        with st.popover(' 延伸问题....'):
            button_names = st.session_state["further_questions_lists"]
            col1, col2, col3 = st.columns([1,1,1])
            buttons = [col1.button(name,on_click=button_clicked,args=(name,), use_container_width=True) for name in button_names]


    if 'retrieval_answer' in st.session_state:
        st.session_state['messages'].append({"role":"assistant","avatar":"🤖","text":st.session_state['retrieval_answer']})
        with st.chat_message("assistant",avatar='🤖'):
            st.write_stream(stream_data(st.session_state['retrieval_answer']))

