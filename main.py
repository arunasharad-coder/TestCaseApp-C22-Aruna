%%writefile app.py
import os
from getpass import getpass
from typing import Annotated, Sequence, TypedDict, Callable
import operator

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph, END

# Set OpenAI API key from environment variable (for Streamlit deployment)
# In a production environment, use Streamlit secrets or other secure methods.
if "OPENAI_API_KEY" not in os.environ:
    st.warning("OPENAI_API_KEY environment variable not found. Please set it for the app to function.")

# Initialize the language model
llm = ChatOpenAI(model="gpt-5") # Remember to change this to a valid model like 'gpt-4o-mini' or 'gpt-3.5-turbo' for deployment

# Define AgentState
class AgentState(TypedDict):
    user_input: str
    test_cases: Annotated[list[str], operator.add]
    reflection: str

# Define the output format for test cases
class TestCases(BaseModel):
    test_cases: list[str] = Field(description="List of diverse and challenging test questions")

# Create a parser for the TestCases model
test_cases_parser = PydanticOutputParser(pydantic_object=TestCases)

# Create the prompt template for the designer node
test_case_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert at generating comprehensive test questions for a given user input. Generate a list of diverse and challenging test questions based on the user's query.\n{format_instructions}"),
        ("human", "{user_input}"),
    ]
).partial(format_instructions=test_cases_parser.get_format_instructions())

# Chain the prompt, LLM, and parser
test_case_generator = test_case_prompt | llm | test_cases_parser

def designer_node(state: AgentState):
    """
    Generates test questions based on user input and updates the AgentState.
    """
    user_input = state["user_input"]
    st.write(f"---GENERATING TEST CASES FOR: {user_input}---")
    generated_test_cases_obj = test_case_generator.invoke({"user_input": user_input})
    generated_test_cases = generated_test_cases_obj.test_cases
    return {"test_cases": generated_test_cases} # Return all generated test cases

# Create the prompt template for the reviewer node
reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an impartial reviewer. Your task is to critically assess the generated test cases based on the original user query. Provide constructive feedback on their quality, diversity, and relevance. If the test cases are good, state that they are good. If there are areas for improvement, clearly outline them. Focus on how well the test cases cover different aspects of the user's query and their potential to thoroughly test a system.\n\nOriginal user query: {user_input}\nGenerated test cases: {test_cases}"),
        ("human", "Review the above test cases."),
    ]
)

# Chain the prompt with the LLM
reviewer_chain = reviewer_prompt | llm

def reviewer_node(state: AgentState):
    """
    Generates a reflection on the quality of the test cases.
    """
    user_input = state["user_input"]
    test_cases = state["test_cases"]
    st.write(f"---REVIEWING TEST CASES FOR: {user_input} with {len(test_cases)} cases---")
    reflection = reviewer_chain.invoke({"user_input": user_input, "test_cases": test_cases}).content
    return {"reflection": reflection}

# Create a prompt for grading the reflection
grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a meticulous grader. Your task is to evaluate the provided reflection on test cases. Based on the feedback, decide if the test cases need further refinement or if they are satisfactory.\nRespond with 'continue' if refinement is needed (e.g., if the reflection suggests improvements, indicates lack of diversity, or highlights missing aspects). Respond with 'end' if the test cases are satisfactory and no further changes are required.\nReflection: {reflection}"""),
        ("human", "Does the reflection indicate that the test cases are good enough or need more work? Output 'continue' or 'end'."),
    ]
)

# Chain the grader prompt with the LLM
grader_chain = grader_prompt | llm

def grade_test_cases(state: AgentState):
    """
    Evaluates the reflection to decide whether to continue or end.
    """
    reflection = state["reflection"]
    st.write(f"---GRADING REFLECTION---")
    decision = grader_chain.invoke({"reflection": reflection}).content.strip().lower()
    st.write(f"---DECISION: {decision} ---")
    if "continue" in decision:
        return "designer"
    else:
        return END

# Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("designer", designer_node)
graph.add_node("reviewer", reviewer_node)
graph.set_entry_point("designer")
graph.add_edge("designer", "reviewer")
graph.add_conditional_edges(
    "reviewer",
    grade_test_cases,
    {
        "designer": "designer",
        END: END
    },
)

# Compile the graph
app = graph.compile()


# Streamlit UI
st.set_page_config(page_title="Agentic QA App", layout="wide")
st.title("🧠 Agentic QA App with LangGraph")
st.write("Enter a query below to generate and review test cases using a multi-agent system.")

user_query = st.text_area("Enter your query here:", height=150)

if st.button("Generate Test Cases"):
    if user_query:
        st.subheader("Generating Test Cases...")
        initial_state = {"user_input": user_query, "test_cases": []}

        # Run the LangGraph application
        final_state = app.invoke(initial_state)

        st.session_state.all_test_cases = final_state["test_cases"]
        st.session_state.current_page = 0
        st.session_state.page_size = 100
        st.session_state.reflection = final_state["reflection"]

    else:
        st.warning("Please enter a query to generate test cases.")

if "all_test_cases" in st.session_state:
    all_test_cases = st.session_state.all_test_cases
    current_page = st.session_state.current_page
    page_size = st.session_state.page_size

    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, len(all_test_cases))

    st.subheader(f"Generated Test Cases ({start_idx + 1}-{end_idx} of {len(all_test_cases)}):")
    for i, tc in enumerate(all_test_cases[start_idx:end_idx]):
        st.write(f"{start_idx + i + 1}. {tc}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous Page", disabled=(current_page == 0)):
            st.session_state.current_page -= 1
            st.experimental_rerun()
    with col2:
        if st.button("Next Page", disabled=(end_idx == len(all_test_cases))):
            st.session_state.current_page += 1
            st.experimental_rerun()

    st.write("### Final Reflection:")
    st.write(st.session_state.reflection)

