from IPython.display import Markdown, HTML, display
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine
import pandas as pd
import urllib

df = pd.read_csv("book4.csv").fillna(value = 0)

username = 'postgres'
password = 'Your Password'
host = 'localhost'
port = '5432'  # Default PostgreSQL port
database = 'AI_AGENTS'

# Create an engine to connect to the SQLite database
# SQLite only requires the path to the database file
encoded_password = urllib.parse.quote_plus(password)
engine = create_engine(f'postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}')
df.to_sql(
    'all_states_history',  # Table name
    con=engine,
    if_exists='replace',  # Replace table if it exists
    index=False  # Don't write DataFrame index as a column
)


MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.

## Tools:

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [death]
FROM covidtracking 
WHERE state = 'TX' AND date LIKE '2020%'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 people who died of covid in Texas in 2020.

Explanation:
I queried the `covidtracking` table for the `death` column where the state
is 'TX' and the date starts with '2020'. The query returned a list of tuples
with the number of deaths for each day in 2020. To answer the question,
I took the sum of all the deaths in the list, which is 27437.
I used the following query

```sql
SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
```
===> End of Example

"""


model = AzureChatOpenAI(
    openai_api_version="",
    azure_deployment="",
    azure_endpoint="",
    api_key=""
)

db = SQLDatabase.from_uri(f'postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}')
toolkit = SQLDatabaseToolkit(db=db, llm=model)


QUESTION = "query the table all_states_history and give me the maximum value from each column"

agent_executor_SQL = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True
)


agent_executor_SQL.invoke(QUESTION)
