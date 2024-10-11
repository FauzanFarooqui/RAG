from llm import llm
#from graph import graph
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
# Create a movie chat chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter

from uuid import uuid4
from langchain_community.chat_message_histories import ChatMessageHistory

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

memory = ChatMessageHistory() #ephemeral memory for the current session

def get_memory(session_id):
    return memory

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()
# Create a set of tools
from langchain.tools import Tool

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    )
]

# Create chat history callback
#from langchain_community.chat_message_histories import Neo4jChatMessageHistory

#def get_memory(session_id):
#    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)
# Create the agent
# from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain import hub

template = """
TOOLS:

------

You have access to the following tools:

{tools}

CHOOSE ONE FROM {tool_names} for the "Action".

Action: General Chat
Action Input: {input}
Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No

Final Answer: [your response here]



```

Begin!



New input: {input}

{agent_scratchpad}

"""

# agent_prompt = ChatPromptTemplate.from_template(template)

# #agent_prompt = hub.pull("hwchase17/react-chat")
# agent = create_react_agent(llm, tools, agent_prompt)
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True
#     )




from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from llm import retriever

system_prompt = (
    """You are a 5G assistant for question-answering tasks on the NRUP ETSI TS Specification. 
    Use the following pieces of retrieved context to answer the question:
    \n\n Context:
    {context}
    \n
    Use the following chat history to refer back to the conversation:
    {chat_history}"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, RunnablePassthrough.assign(chat_history = itemgetter("chat_history")) | question_answer_chain)

chat_mem = RunnableWithMessageHistory(
    rag_chain,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
from utils import get_session_id


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_mem.invoke(
        {"input": user_input},
        {"configurable": {"session_id": SESSION_ID}},)

    return response['answer']
