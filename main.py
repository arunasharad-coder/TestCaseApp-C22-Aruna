import os
import pandas as pd
import io
import streamlit as st
from typing import TypedDict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
if "OPENAI_API_KEY" not in os.environ:
    st.warning("Please set OPENAI_API_KEY in your environment variables.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Agent State ---
class AgentState(TypedDict):
    user_input: str
    test_cases: list[str] 
    reflection: str

# --- Output Model ---
class TestCases(BaseModel):
    test_cases: list[str] = Field(
        description="List of exactly 5 manual test cases in the 1-5 navigation format."
    )

test_cases_parser = PydanticOutputParser(pydantic_object=TestCases)

# --- Nodes ---
test_case_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a QA Lead. Generate EXACTLY 5 test cases. \n"
               "FORMAT:\n1. Go to -[URL]\n2. Click [Element]\n3. Enter [Data]\n4. Click [Button]\n5. Validate -[Result]\n\n"
               "{format_instructions}"),
    ("human", "{user_input}"),
]).partial(format_instructions=test_cases_parser.get_format_instructions())

test_case_generator = test_case_prompt | llm | test_cases_parser

def designer_node(state: AgentState):
    generated = test_case_generator.invoke({"user_input": state["user_input"]})
    return {"test_cases": generated.test_cases[:5]}

def reviewer_node(state: AgentState):
    # Simple pass-through for this version to keep it fast
    return {"reflection": "Format looks good."}

# --- Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("designer", designer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.set_entry_point("designer")
workflow.add_edge("designer", "reviewer")
workflow.add_edge("reviewer", END)
app = workflow.compile()

# --- CSV Helper Function ---
def convert_to_csv(test_list):
    jira_data = []
    for i, tc in enumerate(test_list):
        lines = tc.splitlines()
        
        # We assume line 5 is the "Validate" line based on your 1-5 format
        steps = "\n".join(lines[:4])  # Steps 1 to 4
        expected_result = lines[4] if len(lines) >= 5 else "Result not specified"
        
        jira_data.append({
            "Summary": f"Test Case {i+1}: " + (lines[0] if lines else "Navigation"),
            "Description": steps,
            "Expected Result": expected_result, # New Column!
            "Issue Type": "Test",
            "Priority": "Medium"
        })
    
    df = pd.DataFrame(jira_data)
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit UI ---
st.set_page_config(page_title="QA Test Case Gen", layout="centered")
st.title("📋 QA Step Generator")

query = st.text_input("Describe the feature to test:")

if st.button("Generate 5 Cases", type="primary"):
    if query:
        with st.spinner("Processing..."):
            results = app.invoke({"user_input": query, "test_cases": []})
            st.session_state.final_cases = results["test_cases"]
    else:
        st.error("Please enter a requirement.")

if "final_cases" in st.session_state:
    st.subheader("Results")
    for tc in st.session_state.final_cases:
        st.code(tc, language="text")
    
    # Download Button
    csv_data = convert_to_csv(st.session_state.final_cases)
    st.download_button(
        label="📥 Download Test Cases (CSV)",
        data=csv_data,
        file_name="test_cases.csv",
        mime="text/csv",
    )
