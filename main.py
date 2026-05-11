import os
import pandas as pd
import io
import streamlit as st
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
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
    test_count: int
    test_type: str
    requires_auth: bool

# --- Output Model ---
class TestCase(BaseModel):
    steps: str = Field(description="The 4 navigation steps (Go to, Click, Enter, Click)")
    expected_result: str = Field(description="The 5th step starting with 'Validate -'")
    selectors: str = Field(description="CSS selectors or IDs for elements in the steps (e.g., #login-btn, .search-bar)")

class TestSuite(BaseModel):
    test_cases: list[TestCase] = Field(description="List of test cases as requested")

test_cases_parser = PydanticOutputParser(pydantic_object=TestSuite)

# --- Prompts & Chains ---
test_case_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior QA Engineer. Create exactly {test_count} test cases of type: {test_type}.

    Test type guidance:
    - Positive (Happy Path): Standard user flows where everything works as expected.
    - Negative: Invalid inputs, wrong credentials, missing required fields, error states.
    - Edge Cases: Boundary values, unusual inputs (empty, max length, special characters, unicode).
    - Accessibility: Keyboard navigation, screen reader compatibility, ARIA labels, focus order.
    - Mixed: A balanced combination of positive, negative, and edge cases across the suite.

    {auth_context}

    For each test case, provide:
    1. 'steps': 4 numbered navigation steps appropriate for the test type.
    2. 'expected_result': A single 'Validate -' statement describing the expected outcome.
    3. 'selectors': Specific CSS selectors for the elements used (e.g., #login-btn, .search-bar).

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

# --- Selenium Java + TestNG chain ---
selenium_java_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior SDET specializing in Selenium Java with TestNG.

Convert the provided manual test steps, expected result, and selectors into a clean, production-quality Selenium Java test class using TestNG.

REQUIREMENTS:
1. Generate a complete Java class — not just the test method.
2. Use TestNG annotations: @BeforeMethod (setup), @Test (the test), @AfterMethod (teardown).
3. The @Test annotation should include: groups (e.g., {{"smoke"}}) and a meaningful priority value.
4. Use proper Selenium WebDriver imports and ChromeDriver for setup.
5. Use Selenium's By selector strategies — match the selectors provided:
   - If selector starts with #, use By.id()
   - If selector starts with ., use By.cssSelector()
   - If selector looks like an XPath, use By.xpath()
6. Use TestNG's Assert class for verifications (Assert.assertTrue, Assert.assertEquals, etc.) with descriptive failure messages.
7. Wrap the validation step in a proper Assert statement matching the "Validate -" expected_result.
8. Include explicit waits using WebDriverWait + ExpectedConditions where appropriate (instead of Thread.sleep).
9. The class name should reflect the test purpose (e.g., GoogleSearchTest, LoginValidationTest).
10. Include a brief Javadoc comment above the @Test method explaining what it verifies.

DO NOT include:
- Maven pom.xml or build configuration
- Multiple test methods in one class (one test class = one test for now)
- Explanatory text before or after the code
- Markdown code fences (the UI will format the code)

Output ONLY the Java class code, starting from `package` or `import` statements."""),
    ("human", "Steps: {steps}\nExpected Result: {expected_result}\nSelectors: {selectors}"),
])

selenium_java_chain = selenium_java_prompt | llm

# --- Nodes ---
def designer_node(state: AgentState):
    user_req = state["user_input"].lower()
    
    if len(user_req) < 10:
        return {"reflection": "⚠️ **Please enter a relevant testing question.** (Example: 'Go to `https://google.com` and test the search button')"}
    
    search_context = ""
    if "http" in user_req or ".com" in user_req:
        st.write("🔍 Searching for real-time site details to prevent hallucinations...")
        search_results = search_tool.invoke({"query": f"header links and dropdown menus on {user_req}"})
        search_context = f"\nReal-time search context: {search_results}"
    
    # Build auth context based on user selection
    if state.get("requires_auth"):
        auth_context = (
            "IMPORTANT: This site requires authentication. Generate test cases that ASSUME the user is "
            "already logged in (do NOT include login steps). Tests will run with a pre-authenticated "
            "browser session via Playwright's storage state pattern."
        )
    else:
        auth_context = "No authentication required for this feature."
    
    try:
        prompt_input = f"{state['user_input']}{search_context}"
        generated = test_case_generator.invoke({
            "user_input": prompt_input,
            "test_count": state.get("test_count", 5),
            "test_type": state.get("test_type", "Positive (Happy Path)"),
            "auth_context": auth_context,
        })
        # Slice to exact count requested (in case AI generates extra)
        count = state.get("test_count", 5)
        return {"test_cases": generated.test_cases[:count], "reflection": "Passed Initial Generation"}
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
st.set_page_config(page_title="QAGenie", layout="centered")

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
st.markdown("""
# QAGenie
###### AI-powered test case generation for QA teams
""")
# --- NEW: Configuration controls ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    test_count = st.slider(
        "Number of test cases",
        min_value=3,
        max_value=10,
        value=5,
        help="How many test cases should the AI generate?"
    )

with col2:
    test_type = st.selectbox(
        "Test type",
        ["Positive (Happy Path)", "Negative", "Edge Cases", "Accessibility", "Mixed"],
        index=0,
        help="What kind of test cases do you want?"
    )

with col3:
    requires_auth = st.checkbox(
        "🔐 Site requires login",
        value=False,
        help="Check if the feature is behind a login wall"
    )
# Framework selector (full-width row below the 3-column controls)
framework = st.selectbox(
    "🛠️ Output framework",
    [
        "Playwright TypeScript",
        "Selenium Java + TestNG",
    ],
    index=0,
    help="Choose the framework and language for the generated test scripts"
)

# Show auth guidance only when checked
if requires_auth:
    st.info(
        "ℹ️ **Authentication assumed.** Generated tests will assume the user is already logged in. "
        "We'll include a setup script using Playwright's storage state pattern. "
        "You'll need to run that setup once before your tests can run."
    )

query = st.text_input(
    "Describe the feature to test:",
    placeholder="e.g. Test search functionality on google.com"
)

# 3. MAIN TRIGGER (LangGraph Execution)
if st.button(f"Generate {test_count} Cases", type="primary"):
    if query:
        with st.spinner("Factory is running QC..."):
            results = app.invoke({
                "user_input": query,
                "test_cases": [],
                "test_count": test_count,
                "test_type": test_type,
                "requires_auth": requires_auth,
            })
            
            # GUARDRAIL CHECK: Catch errors from the Designer or Reviewer nodes
            if "⚠️" in results.get("reflection", "") or "Error" in results.get("reflection", ""):
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
            # Dynamic tab label based on framework selection
            script_tab_label = f"🤖 {framework} Script"
            tab_manual, tab_auto = st.tabs(["📝 Manual Steps", script_tab_label])
                        
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
                if st.button(f"Generate {framework} Code for TC {i+1}", key=f"gen_{i}"):
                    with st.spinner("Writing script..."):
                        # Pick the right chain based on framework selection
                        if framework == "Selenium Java + TestNG":
                            chain = selenium_java_chain
                        else:  # Default: Playwright TypeScript
                            chain = playwright_chain
                        
                        code_out = chain.invoke({
                            "steps": tc.steps,
                            "expected_result": tc.expected_result,
                            "selectors": tc.selectors
                        })
                        st.session_state[f"pw_code_{i}"] = code_out.content
                
                # Show code and download button if script exists in session state
                if f"pw_code_{i}" in st.session_state:
                    # Pick syntax highlighting based on framework
                    code_language = "java" if framework == "Selenium Java + TestNG" else "typescript"
                    st.code(st.session_state[f"pw_code_{i}"], language=code_language)
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
