import streamlit as st
import openai
import sqlalchemy
import pandas as pd 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os 
import re

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
 
with open('sample_graph.py', 'r') as file:
  sample_graph = file.read()

#chatGPT settings

model = 'gpt-4'
temperature = 0.2


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
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates natural language questions into complete SQL queries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=temperature,
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

def sanitize_text(text):
    # Replace unwanted characters
    clean_text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    # Additional cleaning can be added here
    return clean_text


# Function to generate a graph and insights using GPT
def get_visualization_and_insights(dataframe):
    # Convert DataFrame to CSV string for GPT processing
    csv_data = dataframe.to_csv(index=False)

    prompt = f"The following is a dataset from an SQL query. Based on the data and provide 1 or 2 insights:\n{csv_data}."
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a data analyst who can analyze datasets and generate meaningful visualizations and insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=temperature,
    )
    
    analysis = response['choices'][0]['message']['content'].strip()
    
    return analysis

def generate_graph_code(dataframe):
    csv_data = dataframe
    print(csv_data)
    prompt = f"""The following is a dataset from an SQL query.Generate Python code to plot a suitable graph for Streamlit:\n\n{csv_data}\n\n.
                 Return on;ly the python code. Don't add any pretext or comments.
                 Don't add ```python. 
                 Use {sample_graph} as an example. 
                 The DataFrame 'csv_data' is already loaded. 
                 Don't try to load a file. The data is already loaded in {csv_data}.
                 Do not attempt to load any CSV files or reference any variables not defined.
                 Always sort the data in descending order.
                 Don't save the plot as a PNG file and display the image. Directly display the plot using st.pyplot().
                 Never use plt.barh().
                 Use rounded values for labels in y axis using plt.yticks().
                 Adjust the figsize based on the length of the x axis labels. 
                 Rotate the x-axis labels to 45 for better readability. 
                 Don't display any warnings. 
                 Always use 'blue' for bar charts.
                 If the dataset is too small just write code to display "Not enough data"
                 """
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a data analyst who can analyze datasets and generate meaningful visualizations and insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5000,
        temperature=temperature,
    )

    # Extract the generated code

    full_text = response['choices'][0]['message']['content'].strip()
    code_match = re.search(r'```python\n(.*?)```', full_text, re.DOTALL)
    
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Fallback if no code block is found
        code = full_text
    
    print(code)
    return code, csv_data


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

            # Clear previous results, insights, and visualizations
            st.session_state.sql_result = None
            st.session_state.insights = None
    else:
        st.warning("Please enter a question.")
st.markdown('</div>', unsafe_allow_html=True)

# Display the generated SQL query if it exists
if 'sql_query' in st.session_state:
    # Calculate the height of the text area based on the number of lines in the SQL query
    sql_lines = st.session_state.sql_query.count('\n') + 5
    height = min(max(100, sql_lines * 20), 700)  # Adjust height based on number of lines, with limits

    # Editable text area for the SQL query
    updated_sql_query = st.text_area("Generated SQL Query:", value=st.session_state.sql_query, height=height)

    # Update the session state with the edited SQL query
    st.session_state.sql_query = updated_sql_query

# Container for the "Execute SQL" button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if 'sql_query' in st.session_state and st.button("Execute SQL"):
    with st.spinner('Executing SQL query...'):
        sql_query = st.session_state.sql_query
        result = execute_sql_query(sql_query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.session_state.sql_result = result  # Store the SQL result in session state
        else:
            st.session_state.sql_result = None  # Reset the result if no data or error

# Display the result if it exists and is not empty
if 'sql_result' in st.session_state and st.session_state.sql_result is not None:
    display_result = st.session_state.sql_result.reset_index(drop=True)
    st.dataframe(display_result, hide_index=True)

# Show "Visualize and Analyze" button if results are available
if 'sql_result' in st.session_state:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Visualize and Analyze"):
        with st.spinner('Generating visualization and insights...'):
            # Generate insights using GPT
            analysis = get_visualization_and_insights(st.session_state.sql_result)
            st.subheader("Insights")
            sanitized_insights = sanitize_text(analysis)
            st.markdown(sanitized_insights)
            # Generate the graph code using the dataset description
            graph_code, csv_data = generate_graph_code(st.session_state.sql_result)
            print(csv_data)

            # Print the generated code (optional, for debugging)
            print(graph_code)
            st.subheader("Visualization")

            # Execute the generated code
            exec(graph_code)
    st.markdown('</div>', unsafe_allow_html=True)