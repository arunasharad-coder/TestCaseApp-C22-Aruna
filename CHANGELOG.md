# QAGenie Changelog
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