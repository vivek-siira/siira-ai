from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import os

# Load environment variables from .env file
load_dotenv()

# Set the name of LLM model and temperature 
model_name = "gpt-4o"
temperature_value = 0.2

def init_database() -> SQLDatabase:
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    database = os.getenv('DB_DATABASE')

    # Update the connection URI for SQL Server
    db_uri = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        
        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        
        For example:
        Question: which 3 artists have the most tracks?
        SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
        Question: Name 10 artists
        SQL Query: SELECT Name FROM Artist LIMIT 10;
        
        Your turn:
        
        Question: {question}
        SQL Query:
        """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model=model_name, temperature=temperature_value)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model=model_name, temperature=temperature_value)
    
    try:
        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
    except Exception as e:
        # If an error occurs, provide a natural language response
        fallback_template = """
            You are a knowledgeable assistant. Answer the user's question based on the conversation history and your knowledge.
            
            Conversation History: {chat_history}
            User question: {question}"""
        
        fallback_prompt = ChatPromptTemplate.from_template(fallback_template)
        
        fallback_chain = (
            RunnablePassthrough
            | fallback_prompt
            | llm
            | StrOutputParser()
        )
        
        response = fallback_chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })

    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

st.set_page_config(page_title="Chat with Siira AI", page_icon=":speech_balloon:")

st.title("Chat with Siira AI")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using SQL Server. Connect to the database and start chatting.")
    
    st.text_input("Host", value=os.getenv('DB_HOST'), key="Host")
    st.text_input("Port", value=os.getenv('DB_PORT'), key="Port")
    st.text_input("User", value=os.getenv('DB_USER'), key="User")
    st.text_input("Password", type="password", value=os.getenv('DB_PASSWORD'), key="Password")
    st.text_input("Database", value=os.getenv('DB_DATABASE'), key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database()
            st.session_state.db = db
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
