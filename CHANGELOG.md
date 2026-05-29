# QAGenie Changelog

## v2.4 — Thursday, May 21, 2026 — Phase 3 + 3.5 + 4a bundled deploy

**Status:** Deployed to production in one push on 2026-05-21 afternoon. Three phases of work shipped together after Phase 3 sat locally for a week.

### Shipped

**Phase 3 — Playwright Python (added 3rd framework)**
- New `playwright_python_prompt` + `playwright_python_chain` in `main.py`
- Senior-quality prompt with 10 requirements + DO NOTs:
  - Self-contained pytest test functions (no separate fixtures)
  - `sync_playwright()` context manager pattern
  - pytest markers (`@pytest.mark.smoke`)
  - Playwright's `expect()` for browser state, `assert` for general checks
  - Semantic locators (`get_by_role()`, `get_by_placeholder()`)
  - No `time.sleep()`, no `print()`, no unittest-style assertions, no markdown fences
- Framework dropdown extended from 2 → 3 options
- Chain routing extended with `elif` branch
- Syntax highlighting + file extension logic refactored from ternary to if/elif/else for cleaner 3+ framework support

**Phase 3.5 — Polish pass (B2 + C1/C2/C3)**
- **B2 fixed:** Download button label + `file_name` now dynamic per framework (uses `file_ext` derived from selected framework instead of hardcoded `.spec.ts`)
- **C1/C2/C3 fixed:** Sidebar copy generalized for multi-framework — no longer references "Playwright scripts" exclusively. Now reads "test scripts" / "across multiple frameworks"
- **B1 deferred to Phase 5** — expander/tab state resets after Generate Code click. Approach C stalled on nested `with` indentation issues; Streamlit also has no native API to control tab state. Decision: Phase 5 will restructure the display loop entirely (replace `st.tabs()` with a controllable pattern) and B1 dissolves as a byproduct.
- **L2 deferred to Phase 5** — ✅ marker on TC headers after code generation. Bundled with B1 attempt; same deferral logic.

**Phase 4a — Selenium Python (4th framework)**
- New `selenium_python_prompt` + `selenium_python_chain` in `main.py`
- Came together in ~30 min using the now-mature 10-requirements + DO NOT prompt template
- Senior-quality requirements:
  - Function-based pytest (not class-based)
  - `@pytest.fixture` named `driver` for WebDriver setup/teardown
  - `webdriver-manager` for ChromeDriver setup (avoids manual driver downloads)
  - pytest markers for categorization
  - By.ID / By.CSS_SELECTOR / By.XPATH strategy matching the selectors provided
  - `WebDriverWait` + `expected_conditions` (never `time.sleep()`)
  - Descriptive `assert` failure messages
  - Docstring inside test function
- Framework dropdown extended to 4 options
- Chain routing + syntax highlighting + file extension logic extended for Selenium Python

### Why it matters

QAGenie is now feature-complete for the core multi-framework vision: **4 frameworks, 2 languages, modern + enterprise stacks covered**.
- Playwright TypeScript (modern web)
- Playwright Python (modern web, Python-native teams)
- Selenium Java + TestNG (enterprise, fintech/healthcare/insurance default)
- Selenium Python (data/ML-adjacent QA teams, Python shops with Selenium investment)

Any QA team running any common stack can now generate first-draft test code with QAGenie. No framework switching required at the prompt level — one click in the dropdown.

### Validation

- Verified Playwright Python output: clean pytest test function with `sync_playwright()` context manager, no anti-patterns
- Verified Selenium Python output: proper `@pytest.fixture(driver)`, `webdriver-manager` import, By strategy matching, WebDriverWait usage
- Regression-tested Playwright TypeScript and Selenium Java — both still generate correctly
- Verified B2 fix: Playwright TS download = `.spec.ts`, Playwright Python = `.py`, Selenium Java = `.java`, Selenium Python = `.py`
- Verified C1/C2/C3 fixes: sidebar copy generic across framework selections

### Files changed

`main.py` — extensive:
- 2 new prompt chains (`playwright_python_chain`, `selenium_python_chain`)
- Framework dropdown expanded to 4 options
- Chain selection logic in display loop extended with two new `elif` branches
- `code_language` if/elif/else for 4 frameworks
- `file_ext` if/elif/else for 4 frameworks (B2 fix)
- Sidebar copy generalized (C1/C2/C3)

### Known Issues (carried forward)

- **B1:** Expander/tab state resets after code generation — deferred to Phase 5 restructure
- **L1:** No progress indicators between Generate → Display steps — deferred to Phase 5
- **L2:** ✅ marker on TC headers after successful code generation — deferred to Phase 5
- **Gap 1:** No POM, no framework config, no CI/CD integration — addressed in V3 (Phase 10+)
- **Gap 2:** Selectors are best-guess from feature description, not from observing real app — partial mitigation via Tavily search context for URLs; full fix is V3 territory

### LinkedIn post

**Post 4** drafted same week (~Sun 5/24-Mon 5/25), scheduled and published Tue 2026-05-27 at 10:21 AM CT. Coverage-led + honest-gap framing. First architect-level engagement on the comments (Cristian N.), plus complementary-tool engagement (Keber Flores, `@keber/qa-framework`).

## v2.3 — Friday, May 15, 2026

### Added
- **Playwright Python** as a third output framework (local only, not deployed)
- Playwright Python prompt chain in `main.py` with senior-quality requirements:
  - Self-contained pytest test functions
  - `sync_playwright()` context manager pattern
  - pytest markers (`@pytest.mark.smoke`)
  - Playwright's `expect()` for browser state, `assert` for general checks
  - No `time.sleep()`, no `print()`, no unittest-style assertions
- Framework dropdown extended from 2 to 3 options (Playwright TypeScript, Playwright Python, Selenium Java + TestNG)
- Chain routing logic extended with `elif` branch for Playwright Python
- Syntax highlighting logic converted from ternary to if/elif/else for cleaner 3+ framework support

### Deployed
- v2.2 (Phase 2 — Selenium Java + TestNG) pushed to production after sitting locally since 5/11

### Known Issues
- B1: Expander/tab state resets after code generation (documented in PROJECT_STATE.md)
- B2: Download button always says `.spec.ts` regardless of selected framework
- C1-C3: Sidebar copy still references "Playwright scripts" — needs generalization for multi-framework

## v2.2 — 2026-05-11 — Phase 2: Selenium Java + TestNG output
**Status:** Committed locally — deploy scheduled for Wed/Thu 2026-05-13/14

**Shipped (locally):**
- Framework selector dropdown (Playwright TS, Selenium Java + TestNG)
- Senior-level Selenium Java prompt chain (TestNG annotations, WebDriverWait, Assert with messages, Javadoc)
- Dynamic UI labels (button, tab) match selected framework
- Java syntax highlighting for Java output

**Why it matters:** v1 and v2.1 only generated Playwright TypeScript. Java/Selenium dominates enterprise QA job postings (~50-60% of mid-to-senior listings). With this update, QAGenie produces code for both modern (Playwright TS) and enterprise (Selenium Java + TestNG) stacks.

**Validation:**
- Verified Java output includes @BeforeMethod, @Test, @AfterMethod
- Verified groups + priority + Javadoc generated correctly
- Verified WebDriverWait used (not Thread.sleep)
- Regression-tested: Playwright TypeScript still generates correctly

**Files changed:** main.py (framework dropdown, selenium_java_prompt, selenium_java_chain, dynamic labels, Java syntax highlighting)

**Bug fixed:** Curly-brace escaping in LangChain prompt template (`{"smoke"}` → `{{"smoke"}}`). Classic LangChain gotcha — `{}` is reserved for variable substitution; literal braces need doubling.

**Deploy timing rationale:** Phase 1 LinkedIn post published earlier today (2026-05-11). Posting Phase 2 within 24 hours dilutes engagement. Deploy + post scheduled for Wed/Thu to give Post 1 its full 48-72hr engagement window.

**LinkedIn post:** Drafted in LINKEDIN_POSTS.md, to be published Wed/Thu after deploy.

---

## v2.1 — 2026-05-08 — Phase 1: User Controls

**Shipped:**
- Test count slider (3-10 cases)
- Test type dropdown (Positive / Negative / Edge / Accessibility / Mixed)
- Authentication toggle with Path A guidance (storage state pattern)
- Dynamic button label matching slider value
- Wired all three controls into the LangGraph workflow + AI prompts

**Why it matters:** v1 only generated happy-path Playwright tests with a hardcoded count of 5. Real QA work requires testing negative paths, edge cases, accessibility, and varied counts. v2.1 lets the AI generate the right *type* and *amount* of tests for the situation.

**Validation:**
- Verified count of 3, 4, 5, 6 all produce correct number of cases
- Verified Negative type produces boundary value + input validation tests (single char, 2048 chars, special chars)
- Verified Auth toggle produces tests that skip login steps when checked
- Regression-tested Jira CSV export still works

**Files changed:** main.py (5 sections: schema, prompt template, AgentState, designer_node, button)

**LinkedIn post:** Pending (draft today, post tomorrow)

## v2.0 — 2026-05-08 — Project relaunch
**Shipped:**
- Renamed app to QAGenie
- Cleaned up local dev environment (.venv, .env, gitignore)
- Added python-dotenv for local env var loading
- Created PROJECT_STATE.md, CHANGELOG.md, SESSION_LOG.md tracking files

**Why it matters:** Setting up the foundation for systematic enhancement. From here forward, every phase ships incrementally with a LinkedIn post.

**Files changed:** main.py (load_dotenv), requirements.txt (python-dotenv), .gitignore (env, venv, OS files), README.md (placeholder still)

**LinkedIn post:** Not yet — first post comes after Phase 1 ships