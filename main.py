import os
from typing import Annotated, TypedDict
import operator

import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# Check API Key
if "OPENAI_API_KEY" not in os.environ:
    st.warning(
        "OPENAI_API_KEY environment variable not found. Please set it for the app to function."
    )

# Initialize LLM (use a valid model)
llm = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------
# Agent State
# -----------------------------
class AgentState(TypedDict):
    user_input: str
    test_cases: Annotated[list[str], operator.add]
    reflection: str


# -----------------------------
# Test Case Output Model
# -----------------------------
class TestCases(BaseModel):
    test_cases: list[str] = Field(
        description="List of diverse and challenging test questions"
    )


test_cases_parser = PydanticOutputParser(pydantic_object=TestCases)

# -----------------------------
# Designer Node
# -----------------------------
test_case_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at generating comprehensive test questions "
            "for a given user input.\n{format_instructions}",
        ),
        ("human", "{user_input}"),
    ]
).partial(format_instructions=test_cases_parser.get_format_instructions())

test_case_generator = test_case_prompt | llm | test_cases_parser


def designer_node(state: AgentState):
    user_input = state["user_input"]

    st.write(f"--- GENERATING TEST CASES FOR: {user_input} ---")

    generated = test_case_generator.invoke({"user_input": user_input})

    return {"test_cases": generated.test_cases}


# -----------------------------
# Reviewer Node
# -----------------------------
reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an impartial reviewer. Assess quality, diversity, "
            "and relevance of the generated test cases.\n\n"
            "Original query: {user_input}\n"
            "Generated test cases: {test_cases}",
        ),
        ("human", "Review the above test cases."),
    ]
)

reviewer_chain = reviewer_prompt | llm


def reviewer_node(state: AgentState):
    user_input = state["user_input"]
    test_cases = state["test_cases"]

    st.write(
        f"--- REVIEWING TEST CASES FOR: {user_input} "
        f"with {len(test_cases)} cases ---"
    )

    reflection = reviewer_chain.invoke(
        {"user_input": user_input, "test_cases": test_cases}
    ).content

    return {"reflection": reflection}


# -----------------------------
# Grader Node
# -----------------------------
grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader. Respond 'continue' if refinement is needed "
            "or 'end' if test cases are satisfactory.\n"
            "Reflection: {reflection}",
        ),
        ("human", "Output 'continue' or 'end'."),
    ]
)

grader_chain = grader_prompt | llm


def grade_test_cases(state: AgentState):
    reflection = state["reflection"]

    st.write("--- GRADING REFLECTION ---")

    decision = grader_chain.invoke(
        {"reflection": reflection}
    ).content.strip().lower()

    st.write(f"--- DECISION: {decision} ---")

    if "continue" in decision:
        return "designer"
    return END


# -----------------------------
# Build LangGraph
# -----------------------------
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
        END: END,
    },
)

app = graph.compile()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Agentic QA App", layout="wide")

st.title("🧠 Agentic QA App with LangGraph")
st.write(
    "Enter a query below to generate and review test cases "
    "using a multi-agent system."
)

user_query = st.text_area("Enter your query here:", height=150)

if st.button("Generate Test Cases"):
    if user_query:
        initial_state = {"user_input": user_query, "test_cases": []}

        final_state = app.invoke(initial_state)

        st.session_state.all_test_cases = final_state["test_cases"]
        st.session_state.reflection = final_state["reflection"]
        st.session_state.current_page = 0
        st.session_state.page_size = 100
    else:
        st.warning("Please enter a query.")

if "all_test_cases" in st.session_state:
    all_test_cases = st.session_state.all_test_cases
    current_page = st.session_state.current_page
    page_size = st.session_state.page_size

    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, len(all_test_cases))

    st.subheader(
        f"Generated Test Cases ({start_idx+1}-{end_idx} of {len(all_test_cases)})"
    )

    for i, tc in enumerate(all_test_cases[start_idx:end_idx]):
        st.write(f"{start_idx + i + 1}. {tc}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Page", disabled=(current_page == 0)):
            st.session_state.current_page -= 1
            st.rerun()

    with col2:
        if st.button("Next Page", disabled=(end_idx == len(all_test_cases))):
            st.session_state.current_page += 1
            st.rerun()

    st.write("### Final Reflection:")
    st.write(st.session_state.reflection)
