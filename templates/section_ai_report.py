from utils import *
from graphs import *
'''https://openai.com/enterprise-privacy'''
# brain emojis -> 🧠
import openai
from streamlit_chat import message
import langchain
langchain.verbose = False
openai.api_key = st.secrets["OPENAI_API_KEY"]

def final_page_ai(name_db: str, section: str, name_user: str):
    # need to get the data for each restaurant
    data = Database_Manager(name_db).get_main_db_from_venue()
    st.write(data)
    
    role = st.text_area('Role',
            """
            You are a helpful digital assistant that is helping monitoring a lot of customers reviews.
            You will create a summary considering Positive and Negative points.
            """
        )
    
    question = st.text_area(
            'Specific question',
            "Which patterns do you see in the reviews?")

    # divide the data in chucks depending on the venue
    all_restaurants = data['Reservation: Venue'].unique()

    if st.button('Send', key='send'):
        for res in all_restaurants:
            report_for_restaurant = []
            with st.expander('Restaurant: ' + res):
                # filter data for each restaurant
                data_restaurant = data[data['Reservation: Venue'] == res]
                data_restaurant = data_restaurant['Details']
                # only the non empty ones
                data_restaurant = data_restaurant[data_restaurant != '']
                # make it as a string
                data_restaurant = ' '.join(data_restaurant)
                data_batch = [data_restaurant[i:i+1750] for i in range(0, len(data_restaurant), 1750)]
                # now we need to iterate over the data_batch
                for chunk in data_batch:
                    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": role },
                        {"role": "user", "content": question},
                        {"role": "system", "content": chunk}
                    ]
                    )
                    report_for_restaurant.append(completion.choices[0].message['content'])
                    st.write(completion.choices[0].message['content'])
                # now we need to join all the chunks
                report_for_restaurant_ = ' '.join(report_for_restaurant)

                question_1 = 'Put everything together and generate a in depth report for the restaurant: ' + res
                completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": role },
                            {"role": "user", "content": question_1},
                            {"role": "system", "content": report_for_restaurant_}
                        ]
                        )
            st.write(completion.choices[0].message['content'])

def final_page_ai_regular(name_db :str, section: str, name_user: str):
    import streamlit as st

    from dotenv import load_dotenv
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.llms import HuggingFaceHub
    from htmlTemplates import css, bot_template, user_template
    
    # set openai key
    def get_pdf_text():
        data = Database_Manager(name_db).get_main_db_from_venue()
        data = data[data['Details'] != '']
        values_list = []
        for c in ['Details', 'Reservation: Venue']:
            values = data[c].tolist()
            # transform in strings
            values = [f'{c}: {i}' for i in values]
            values_list.append(values)

        # values_list = [data['Details'].tolist(), data['Reservation: Venue'].tolist()]
        # need to join the two lists element wise
        data = [' '.join(i) for i in zip(*values_list)]
        data = '      '.join(data)
        return data

    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["OPENAI_API_KEY"])
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI(openai_api_key= st.secrets["OPENAI_API_KEY"])
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def handle_userinput(user_question):
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
        except:
            st.info('Load the reviews')

    def main():
        load_dotenv()
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        st.write(css, unsafe_allow_html=True)

        button_question_1 = st.sidebar.button('Generate Report', use_container_width= True)
        button_question_2 = st.sidebar.button('Find the worst reviews', use_container_width= True)
        button_question_3 = st.sidebar.button('Find the best reviews', use_container_width= True)
        
        if button_question_1:
            user_question = 'Generate a in depth report for each restaurant - Highlight positive and negative points (make a numbered list) and talk about ways to improve.'
            handle_userinput(user_question)
        elif button_question_2:
            user_question = 'Find the worst reviews - render a numbered list of the worst reviews'
            handle_userinput(user_question)
        elif button_question_3:
            user_question = 'Find the best reviews - render a numbered list of the best reviews'
            handle_userinput(user_question)
        else:
            user_question = st.chat_input("Ask a question about the reviews:")
            if user_question:
                handle_userinput(user_question)

        c1,c2 = st.sidebar.columns(2)
        with st.sidebar:
            if c1.button("Start Chat", use_container_width= True):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text()
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
            # handle reset
            if c2.button("Reset", use_container_width= True, type="primary"):
                st.session_state.conversation = None
                st.session_state.chat_history = None
    
        # reverse the order of the chat history
        messages = st.session_state.chat_history
        if messages:
            for i, message in enumerate(messages):
                c1,c2 = st.columns(2)
                if i % 2 == 0:
                    c1.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    c2.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
    main()

def final_page_ai_streaming(name_db :str, section: str, name_user: str):
    import streamlit as st
    import time
    import streamlit as st
    from typing import Any, List, Dict
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        HumanMessage,
    )
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import AgentAction, AgentFinish, LLMResult

    
    from dotenv import load_dotenv
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.llms import HuggingFaceHub
    from htmlTemplates import css, bot_template, user_template
    
    class MyStream(StreamingStdOutCallbackHandler):
        def __init__(self, container) -> None:
            super().__init__()
            self.o =  container
            self.container = container
            self.s = ''

        def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
        ) -> None:
            """Run when LLM starts running."""
            del self.o
            self.s = ''
            self.o =  self.container
            
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            """Run on new LLM token. Only available when streaming is enabled."""
            # only the last token
            self.s += token
            self.o.write(user_template.replace(
                            "{{MSG}}", self.s), unsafe_allow_html=True)
            time.sleep(0.05)

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            """Run when LLM ends running."""
            self.s = ''    

    # set openai key
    def get_pdf_text():
        data = Database_Manager(name_db).get_main_db_from_venue()
        data = data[data['Details'] != '']
        values_list = []
        for c in ['Details', 'Reservation: Venue']:
            values = data[c].tolist()
            # transform in strings
            values = [f'{c}: {i}' for i in values]
            values_list.append(values)

        # values_list = [data['Details'].tolist(), data['Reservation: Venue'].tolist()]
        # need to join the two lists element wise
        data = [' '.join(i) for i in zip(*values_list)]
        data = '      '.join(data)
        return data

    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["OPENAI_API_KEY"])
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(vectorstore, llm):
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def handle_userinput(user_question):
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
        except:
            st.info('Load the reviews')

    def main():
        load_dotenv()
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        st.write(css, unsafe_allow_html=True)

        button_question_1 = st.sidebar.button('Generate Report', use_container_width= True)
        button_question_2 = st.sidebar.button('Find the worst reviews', use_container_width= True)
        button_question_3 = st.sidebar.button('Find the best reviews', use_container_width= True)
        
        if button_question_1:
            user_question = 'Generate a in depth report for each restaurant - Highlight positive and negative points (make a numbered list) and talk about ways to improve.'
            handle_userinput(user_question)
        elif button_question_2:
            user_question = 'Find the worst reviews - render a numbered list of the worst reviews'
            handle_userinput(user_question)
        elif button_question_3:
            user_question = 'Find the best reviews - render a numbered list of the best reviews'
            handle_userinput(user_question)
        else:
            user_question = st.chat_input("Ask a question about the reviews:")
            if user_question:
                handle_userinput(user_question)

        # reverse the order of the chat history
        messages = st.session_state.chat_history
        if messages:
            for i, message in enumerate(messages[::-1]):
                if i % 2 != 0:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
        
        container = st.empty()
        llm = ChatOpenAI(streaming=True, callbacks=[MyStream(container)], openai_api_key= st.secrets["OPENAI_API_KEY"])

        c1,c2 = st.sidebar.columns(2)
        if c1.button("Load Reviews", use_container_width= True):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text()
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                        vectorstore, llm)
        # handle reset
        if c2.button("Reset", use_container_width= True, type="primary"):
            st.session_state.conversation = None
            st.session_state.chat_history = None

    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''

    st.components.v1.html(js)
    main()

def final_page_ai(name_db :str, section: str, name_user: str):
    if st.sidebar.toggle('Use Streaming', value=False):
        final_page_ai_streaming(name_db, section, name_user)
    else:
        final_page_ai_regular(name_db, section, name_user)
        