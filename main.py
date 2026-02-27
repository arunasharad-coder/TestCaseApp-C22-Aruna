import os
import pandas as pd
import io
import streamlit as st
from typing import TypedDict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# --- NEW ---
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
if "OPENAI_API_KEY" not in os.environ:
    st.warning("Please set OPENAI_API_KEY in your environment variables.")
# --- NEW ---
if "TAVILY_API_KEY" not in os.environ:
    st.warning("Please set TAVILY_API_KEY in your environment variables.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# --- NEW ---
search_tool = TavilySearchResults(max_results=2)

# --- Agent State ---
class AgentState(TypedDict):
    user_input: str
    test_cases: list[object] 
    reflection: str

# --- Output Model ---
class TestCase(BaseModel):
    steps: str = Field(description="The 4 navigation steps (Go to, Click, Enter, Click)")
    expected_result: str = Field(description="The 5th step starting with 'Validate -'")
    selectors: str = Field(description="CSS selectors or IDs for elements in the steps (e.g., #login-btn, .search-bar)")

class TestSuite(BaseModel):
    test_cases: list[TestCase] = Field(description="List of exactly 5 test cases")

test_cases_parser = PydanticOutputParser(pydantic_object=TestSuite)

# --- Prompts & Chains ---
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

playwright_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Playwright automation expert. 
    Convert these manual steps and selectors into a clean TypeScript Playwright test script. 
    Use 'test' and 'expect' syntax."""),
    ("human", "Steps: {steps}\nResult: {expected_result}\nSelectors: {selectors}"),
])

playwright_chain = playwright_prompt | llm

# --- Nodes ---
def designer_node(state: AgentState):
    user_req = state["user_input"].lower()
    
    if len(user_req) < 10:
        # UPDATE THIS LINE:
        return {"reflection": "⚠️ **Please enter a relevant testing question.** (Example: 'Go to `https://google.com` and test the search button')"}
    
    # --- UPDATED: Use Tavily to get real-time site structure ---
    search_context = ""
    if "http" in user_req or ".com" in user_req:
        st.write("🔍 Searching for real-time site details to prevent hallucinations...")
        # Ask Tavily specifically for header links and dropdown menus
        search_results = search_tool.invoke({"query": f"header links and dropdown menus on {user_req}"})
        search_context = f"\nReal-time search context: {search_results}"
    
    # 2. Second Line of Defense: Try/Except for Parsing
    try:
        # --- MODIFIED: Include search context in the prompt ---
        prompt_input = f"{state['user_input']}{search_context}"
        generated = test_case_generator.invoke({"user_input": prompt_input})
        return {"test_cases": generated.test_cases[:5], "reflection": "Passed Initial Generation"}
    except Exception as e:
        return {
            "test_cases": [], 
            "reflection": "Error: The AI couldn't turn that input into a test case. Please try a specific feature description."
        }
        
def reviewer_node(state: AgentState):
    cases = state.get("test_cases", [])
    if not cases:
        return {"reflection": "Error: No cases generated."}
    
    for i, tc in enumerate(cases):
        if "Validate -" not in tc.expected_result:
            return {"reflection": f"TC {i+1} failed: Missing 'Validate -'"}
        if not tc.selectors or len(tc.selectors) < 2:
            return {"reflection": f"TC {i+1} failed: No selectors found."}
    
    return {"reflection": "Passed QC"}

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
            "Test Step": tc.steps,
            "Expected Result": tc.expected_result,
            "Issue Type": "Test",
            "Status": "To Do",
            "Priority": "Medium"
        })
    df = pd.DataFrame(jira_data)
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit UI ---
st.set_page_config(page_title="QA Test Case Gen", layout="centered")

# 1. SIDEBAR (The "Learning Journey" Version)

with st.sidebar:
    st.header("🐣 My QA Learning Lab")
    st.markdown("""
    Hey! I'm building this to learn how AI can help us in QA. 
    
    **What's happening under the hood?**
    * **The Designer:** An AI agent that tries to think like a manual tester.
    * **The Reviewer:** A second agent that double-checks the work for quality.
    * **Automation:** It even writes the Playwright scripts for me!
    """)
    
    st.divider()
    
    # --- Video Tutorial ---
    st.subheader("📺 Quick Tutorial")
    st.video("https://www.youtube.com/watch?v=nkOLuILaOTo")
    
    st.divider()
    
    # --- Prompt Guidelines Section ---
    st.subheader("💡 Better Prompts = Better Scripts")
    st.markdown("""
    To get accurate Playwright scripts, especially for sites with **many links or dropdowns**:
    
    1.  **Be Explicit:** Instead of "Test the site", say "Hover over 'Services', then click on 'Cloud Solutions'".
    2.  **Define Actions:** Specify if a link should *navigate* to a new page or *open a menu*.
    3.  **Use URLs:** Include the full URL so the AI can search for the correct selectors.
    """)
    
    st.divider()
    st.subheader("⚙️ Built With")
    st.caption("Python | Streamlit | LangChain | GPT-4o-mini")
    
    st.divider()
    if st.button("🔄 Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
# 2. MAIN HEADER
st.title("📋 QA Test Cases Generator")
query = st.text_input("Describe the feature to test:", placeholder="e.g. Test search functionality on google.com")

# 3. MAIN TRIGGER (LangGraph Execution)
if st.button("Generate 5 Cases", type="primary"):
    if query:
        with st.spinner("Factory is running QC..."):
            results = app.invoke({"user_input": query, "test_cases": []})
            
            # GUARDRAIL CHECK: Catch errors from the Designer or Reviewer nodes
            if "Error" in results.get("reflection", "") or "failed" in results.get("reflection", ""):
                st.error(f"🚨 Quality Control Blocked: {results['reflection']}")
            else:
                st.session_state.final_cases = results["test_cases"]
                # Clean up old code sessions to prevent mixing old scripts with new cases
                for key in list(st.session_state.keys()):
                    if key.startswith("pw_code_"):
                        del st.session_state[key]
                st.success("✅ Test suite passed all guardrails!")
    else:
        st.error("Please enter a requirement.")

# 4. DISPLAY LOOP (Manual Steps & Playwright Tabs)
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
                
                # Each button has a unique key using the loop index 'i'
                if st.button(f"Generate Playwright Code for TC {i+1}", key=f"gen_{i}"):
                    with st.spinner("Writing script..."):
                        code_out = playwright_chain.invoke({
                            "steps": tc.steps,
                            "expected_result": tc.expected_result,
                            "selectors": tc.selectors
                        })
                        st.session_state[f"pw_code_{i}"] = code_out.content
                
                # Show code and download button if script exists in session state
                if f"pw_code_{i}" in st.session_state:
                    st.code(st.session_state[f"pw_code_{i}"], language="typescript")
                    st.download_button(
                        label="💾 Download .spec.ts",
                        data=st.session_state[f"pw_code_{i}"],
                        file_name=f"test_{i+1}.spec.ts",
                        mime="text/plain",
                        key=f"dl_{i}" 
                    )

    # 5. GLOBAL DOWNLOAD (Jira CSV)
    st.divider()
    csv_data = convert_to_csv(st.session_state.final_cases)
    st.download_button(
        label="📥 Download All for Jira Import",
        data=csv_data,
        file_name="jira_test_cases.csv",
        mime="text/csv",
        key="global_csv_download"
    )
