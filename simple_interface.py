import streamlit as st
import openai
import sqlalchemy
import pandas as pd 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os 

# Set your OpenAI API key here

load_dotenv()
openai.api_key_path = None
openai.api_key = os.getenv("OPENAI_API_KEY")
USER = os.getenv('DB_USER')
HOST = os.getenv('DB_HOST')
PWD = os.getenv('DB_PWD')
DB = os.getenv('DB_DEFAULT')

# Database connection details
DATABASE_URL = f'postgresql+psycopg2://{USER}:{PWD}@{HOST}/{DB}'

with open('prompt_sql.txt', 'r') as file:
    prompt_sql = file.read()

with open('sample_questions.txt', 'r') as file:
  sample_questions = file.read()
 

# Function to fetch the database schema
def get_db_schema():
    engine = sqlalchemy.create_engine(DATABASE_URL)
    connection = engine.connect()
    schema = {}

    # Fetch table names
    table_names = sqlalchemy.inspect(engine).get_table_names()
    for table in table_names:
        # Fetch column names for each table
        columns = sqlalchemy.inspect(engine).get_columns(table)
        column_names = [column['name'] for column in columns]
        schema[table] = column_names

    connection.close()
    return schema

# Function to generate the SQL query using GPT model
def get_sql_from_gpt(question, schema):
    schema_str = "\n".join([f"Table {table}: {', '.join(columns)}" for table, columns in schema.items()])
    prompt = f"{prompt_sql} Provide only the SQL query without any additional text:\n{schema_str}\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates natural language questions into complete SQL queries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5,
    )
    sql_query = response['choices'][0]['message']['content'].strip()

    # Post-processing to ensure SQL query has a FROM clause
    if "SELECT" in sql_query.upper() and "FROM" not in sql_query.upper():
        sql_query += " FROM default_table"  # Use a default table if no FROM clause is present

    return sql_query

# Function to execute the SQL query and return the result
def execute_sql_query(query):
    engine = sqlalchemy.create_engine(DATABASE_URL)
    connection = engine.connect()
    try:
        result = pd.read_sql_query(query, connection)
    except Exception as e:
        result = f"Error: {e}"
    connection.close()
    return result

# Function to generate a graph and insights using GPT
def get_visualization_and_insights(dataframe):
    # Convert DataFrame to CSV string for GPT processing
    csv_data = dataframe.to_csv(index=False)

    prompt = f"The following is a dataset from an SQL query. Based on the data and provide 1 or 2 insights:\n{csv_data}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst who can analyze datasets and generate meaningful visualizations and insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5,
    )
    
    analysis = response['choices'][0]['message']['content'].strip()
    
    return analysis

# Title of the app
st.title("Siira AI")

# Sidebar with static text
st.sidebar.title("Sample Questions")
st.sidebar.info(sample_questions)

# Initialize the session state for the question if not already present
if 'question' not in st.session_state:
    st.session_state.question = ''

# Input box for the user to enter a question
question = st.text_input("Enter your question:", value=st.session_state.question)

# Update the session state with the current input value
st.session_state.question = question

# Container for buttons
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("Get SQL"):
    if question:
        with st.spinner('Generating SQL query...'):
            schema = get_db_schema()
            sql_query = get_sql_from_gpt(question, schema)
            st.session_state.sql_query = sql_query  # Store the SQL query in session state
    else:
        st.warning("Please enter a question.")
st.markdown('</div>', unsafe_allow_html=True)


# Display the generated SQL query if it exists
if 'sql_query' in st.session_state:
    # Calculate the height of the text area based on the length of the SQL query
    query_length = len(st.session_state.sql_query)
    height = min(max(100, query_length // 50 * 20), 300)  # Adjust the multipliers and limits as needed

    # Editable text area for the SQL query
    updated_sql_query = st.text_area("Generated SQL Query:", value=st.session_state.sql_query, height=height)

    # Update the session state with the edited SQL query
    st.session_state.sql_query = updated_sql_query

if 'sql_query' in st.session_state and st.button("Execute SQL"):
    with st.spinner('Executing SQL query...'):
        sql_query = st.session_state.sql_query
        result = execute_sql_query(sql_query)
        if isinstance(result, pd.DataFrame):
            st.session_state.sql_result = result  # Store the SQL result in session state
        else:
            st.session_state.sql_result = pd.DataFrame({'Result': [str(result)]})  # Store error message in session state

# Display the result if it exists in session state
if 'sql_result' in st.session_state:
    st.table(st.session_state.sql_result)

# Show "Visualize and Analyze" button if results are available
if 'sql_result' in st.session_state:
    if st.button("Visualize and Analyze"):
        analysis = get_visualization_and_insights(st.session_state.sql_result)
        
        # Display the insights
        st.subheader("Insights")
        st.write(analysis)

        # Plotting a basic graph as a placeholder
        st.subheader("Visualization")
        fig, ax = plt.subplots()
        st.session_state.sql_result.plot(ax=ax)  # Basic plot, can be improved
        st.pyplot(fig)
