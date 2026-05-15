# QAGenie — Project State

> **Last updated:** 2026-05-15 (Session 5)
> **Always update this file at end of session.**

## App
- **Name:** QAGenie
- **Repo:** https://github.com/arunasharad-coder/TestCaseApp-C22-Aruna *(to be renamed in Phase 4)*
- **Live URL:** https://testcaseapp-c22-aruna.streamlit.app/ *(to be renamed in Phase 4)*
- **Tagline:** AI-powered test case generation for modern QA teams
- **Project goal:** Land a job in AI-augmented QA / SDET / QA Tools roles

## Current Status
**Phase 3 — Playwright Python (built locally, not deployed)**

- v2.3 (Phase 3) — Playwright Python chain added locally; not yet deployed
- v2.2 (Phase 2) — Selenium Java + TestNG, deployed to production Fri 5/15
- v2.1 — QAGenie branding (deployed earlier)
- v2.0 / v1 — Original Playwright TypeScript output, LangGraph workflow, Tavily search, Jira CSV export

## What's Deployed in Production
- Configurable test count (3-10 via slider)
- Test type selection (Positive / Negative / Edge / Accessibility / Mixed)
- Authentication-aware mode (Path A — assumes session)
- Dynamic button label matches selected count
- QAGenie branding (renamed from "QA Test Cases Generator")
- Tagline visible below title
- Framework selector dropdown (Playwright TypeScript, Selenium Java + TestNG)
- selenium_java_chain with senior-level prompt (TestNG annotations, WebDriverWait, groups, priority, Javadoc)
- Dynamic button label matches selected framework
- Dynamic tab label matches selected framework
- Java syntax highlighting in code viewer
- v1 features: 5 test cases default, LangGraph workflow, Tavily search, Jira CSV export, per-TC code generation, sidebar with tutorial

## What's Committed Locally but NOT Deployed (v2.3)
- Playwright Python chain (senior-quality prompt with 10 requirements + DO NOT section)
- Framework dropdown extended to 3 options (PW TS, PW Python, Selenium Java + TestNG)
- Chain routing logic extended with `elif` for Playwright Python
- `code_language` logic converted from ternary to if/elif/else for 3-framework support
- Python syntax highlighting in code viewer

## Immediate Next Session
**Phase 3.5: UI polish** — Fix bugs and copy before any further prod deploy
- Estimated effort: 60-90 min
- Items: B1 (state collapse bug), B2 (download extension), C1-C3 (sidebar copy)

## Roadmap
1. **Phase 1** ✅ Posted — User controls (slider, type, auth toggle)
2. **Phase 2** ✅ Deployed Fri 5/15 — Selenium Java + TestNG (LinkedIn Post 3 scheduled Mon 5/18)
3. **Phase 3** (current) — Playwright Python (built locally)
4. **Phase 3.5** — UI polish (B1, B2, C1-C3) before next deploy
5. **Phase 4** — Selenium Python + UX upgrade (cards, indicators, empty states) + HF Spaces migration + README polish
6. **Phase 5** — API testing mode (pytest+requests, REST Assured+TestNG)
7. **Phase 6** — Postman Collection export + sample prompts button
8. **Phase 7** — Requirement upload (PDF/docx → test cases)
9. **Phase 8** — Coverage matrix + test history (last 5 runs)

## Locked Decisions
- **Hosting:** Streamlit Cloud now → migrate to HF Spaces in Phase 4 — *migration easier with stable code, not WIP*
- **Auth approach:** Path A (acknowledge limitation, generate auth-aware code with TODO comments) — *honest scope; production teams use storage state pattern anyway*
- **Framework rollout order:** Playwright TS (done) → Selenium Java (done) → Playwright Python (done locally) → Selenium Python (Phase 4) — *Java first for biggest job-market signal*
- **UI framework:** Stay with Streamlit through Phase 8 (decided Fri 5/15 — see UI Framework Decision below)
- **Time budget:** 15 hrs/week
- **Post cadence:** 1 LinkedIn post per shipped phase, every 3-5 days
- **Deploy cadence:** Coordinate deploys with LinkedIn posts — never deploy a major feature without an accompanying announcement

## Known UI Issues (Phase 3.5 backlog)

### Bugs (must fix before next prod deploy)

**B1: Expander/tab state resets after Generate Code click** [HIGH]
- Repro: Click expander on Test Case → pick Playwright Python Script tab → click Generate → state collapses → user must re-expand + re-click tab to see generated code
- Root cause: Streamlit re-runs entire script on button click; expander state not preserved across re-runs
- Fix approach: Use `st.session_state` to track expanded state per test case
- Effort: 30-45 min

**B2: Download button label always says `.spec.ts`** [MEDIUM]
- Wrong extension for Python (`.py`) and Java (`.java`) outputs
- Fix: Make label and file_name dynamic based on `framework` value
- Effort: 10 min

### Copy updates (must fix)

- **C1:** Sidebar "Automation: It even writes the Playwright scripts for me!" — generalize for multi-framework
- **C2:** Sidebar tutorial "To get accurate Playwright scripts..." — same issue
- **C3:** 💡 "Better Prompts = Better Scripts" section — generalize language for multi-framework
- Effort: 15 min total

### Polish (nice to fix, not blocking)

- **L1:** Code block scrolls horizontally on long lines — Streamlit limitation, hard to fully fix
- **L2:** Test case headers don't visually indicate if code has been generated yet (could add ✅ when present)

## UI Framework Decision

**Decision:** Stay with Streamlit through Phase 8.

**Reasoning (Fri 5/15):**
- ~70% of code is already framework-portable (chains, prompts, models, LangGraph workflow)
- Switching frameworks mid-project resets LinkedIn content momentum
- Streamlit is respected in the AI/ML community — not a hiring negative
- Future React rewrite (if desired) is better as a separate portfolio piece, not a replacement
- Migration cost stays low if good habits are followed

**Going forward principles to preserve portability:**
- Keep `st.session_state` reads/writes near UI rendering — don't sprinkle through business logic
- Prefer pure functions (e.g. `generate(steps, expected)`) over Streamlit-coupled functions
- Consider splitting `main.py` into `chains.py`, `models.py`, `workflow.py`, `main.py` (UI) later if migration becomes attractive

## Future Direction / V3 Gaps (post-Phase 8)

QAGenie v2 generates test artifacts but does not yet address these deeper QA workflow problems. These are intentionally out of scope for v2 but represent the most valuable future work.

### Gap 1: Framework integration
QAGenie outputs individual test scripts, not framework-aware code. Real teams need:
- Page Object Model (POM) structure
- Framework config (playwright.config.ts, pytest.ini, testng.xml)
- CI/CD setup (.github/workflows/e2e.yml)
- Adaptation to existing team conventions

### Gap 2: Selector accuracy (the hallucination problem)
The AI doesn't know the actual DOM of the user's app. It generates selectors based on training data and educated guesses. Solutions worth exploring:
- Live DOM inspection (AI navigates to URL and reads actual elements)
- Codebase awareness (AI reads source code to find data-testid attributes)
- Vision-based generation (AI takes screenshots of running app, identifies elements visually)

### Gap 3: Apps without testids
Most apps don't have data-testid attributes. Without them, AI-generated selectors are fragile. A "Selector Audit Mode" could:
- Identify elements lacking testids
- Suggest testid names following naming conventions
- Generate a SELECTOR_MAP.md for the team
- Bridge between "ideal AI-friendly app" and "real-world app"

These three gaps represent v3 territory — substantial work, but the difference between QAGenie being a demo and being a production-grade SDET tool.

## Parking Lot — NOT for v2
- Database persistence
- User accounts / login
- Real Jira/TestRail API integration
- Cypress / Puppeteer support
- Real-time test execution
- Page Object Model generation
- BDD/Gherkin output
- Multi-language UI
- Critic agent / scoring
- Slack/email notifications

## Tech Stack
- Python 3.14, Streamlit 1.57.0
- LangChain, LangGraph
- OpenAI GPT-4o-mini (temperature=0)
- Tavily search (max_results=2)
- python-dotenv