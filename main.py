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
class TestCase(BaseModel):
    steps: str = Field(description="The 4 navigation steps (Go to, Click, Enter, Click)")
    expected_result: str = Field(description="The 5th step starting with 'Validate -'")
    # NEW: This captures the technical IDs/Classes for Playwright
    selectors: str = Field(description="CSS selectors or IDs for elements in the steps (e.g., #login-btn, .search-bar)")

class TestSuite(BaseModel):
    test_cases: list[TestCase] = Field(description="List of exactly 5 test cases")

test_cases_parser = PydanticOutputParser(pydantic_object=TestSuite)
# --- Nodes ---
# Updated Prompt
test_case_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a QA Engineer. Create 5 test cases.
    For each case, provide:
    1. 'steps': 4 numbered navigation steps.
    2. 'expected_result': A single 'Validate -' statement.
    3. 'selectors': Specific CSS selectors for the elements used.
    
    {format_instructions}"""),
    ("human", "{user_input}"),
]).partial(format_instructions=test_cases_parser.get_format_instructions())
test_case_generator = test_case_prompt | llm | test_cases_parser

# The Brain for converting text to code
playwright_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Playwright automation expert. 
    Convert these manual steps and selectors into a clean TypeScript Playwright test script. 
    Use 'test' and 'expect' syntax."""),
    ("human", "Steps: {steps}\nResult: {expected_result}\nSelectors: {selectors}"),
])

playwright_chain = playwright_prompt | llm

def designer_node(state: AgentState):
    # This invokes the generator we just built above
    generated = test_case_generator.invoke({"user_input": state["user_input"]})
    
    # We return the list of TestCase objects (which now includes steps, results, AND selectors)
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
def convert_to_csv(test_suite):
    jira_data = []
    for i, tc in enumerate(test_suite):
        jira_data.append({
            "Summary": f"Test Case {i+1}: Navigation Flow",
            "Test Step": tc.steps,            # Steps 1-4
            "Expected Result": tc.expected_result, # Step 5
            "Issue Type": "Test",
            "Status": "To Do",
            "Priority": "Medium"
        })
    
    df = pd.DataFrame(jira_data)
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit UI ---
st.set_page_config(page_title="QA Test Case Gen", layout="centered")
st.title("📋 QA Test Case Generator")

query = st.text_input("Describe the feature to test:")

# 1. TRIGGER BUTTON (You need this to actually run the AI!)
if st.button("Generate 5 Cases", type="primary"):
    if query:
        with st.spinner("Brainstorming steps and selectors..."):
            # Run your LangGraph
            results = app.invoke({"user_input": query, "test_cases": []})
            # Save to session state so they don't disappear
            st.session_state.final_cases = results["test_cases"]
            # Clear old Playwright code from previous runs
            for key in list(st.session_state.keys()):
                if key.startswith("pw_code_"):
                    del st.session_state[key]
    else:
        st.error("Please enter a requirement.")

# 2. DISPLAY LOGIC (Your code goes here)
if "final_cases" in st.session_state:
    st.subheader("Generated Test Suite")
    
    for i, tc in enumerate(st.session_state.final_cases):
        with st.expander(f"Test Case {i+1}", expanded=False):
            tab_manual, tab_auto = st.tabs(["📝 Manual Steps", "🤖 Playwright Script"])
            
            with tab_manual:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Steps:**")
                    st.code(tc.steps, language="text")
                with col2:
                    st.markdown("**Expected Result:**")
                    st.info(tc.expected_result)
            
            with tab_auto:
                st.markdown("**Target Selectors:**")
                st.caption(tc.selectors)
                
                if st.button(f"Generate Code for TC {i+1}", key=f"gen_btn_{i}"):
                    with st.spinner("Writing Playwright script..."):
                        code_out = playwright_chain.invoke({
                            "steps": tc.steps,
                            "expected_result": tc.expected_result,
                            "selectors": tc.selectors
                        })
                        st.session_state[f"pw_code_{i}"] = code_out.content
                
                if f"pw_code_{i}" in st.session_state:
                    st.code(st.session_state[f"pw_code_{i}"], language="typescript")
                    st.download_button(
                        label="💾 Download .spec.ts",
                        data=st.session_state[f"pw_code_{i}"],
                        file_name=f"test_{i+1}.spec.ts",
                        mime="text/plain",
                        key=f"dl_{i}" 
                    )

    # 3. CSV DOWNLOAD (Keep this at the very bottom)
    csv_data = convert_to_csv(st.session_state.final_cases)
    st.download_button(
        label="📥 Download All for Jira Import",
        data=csv_data,
        file_name="jira_test_cases.csv",
        mime="text/csv",
    )
