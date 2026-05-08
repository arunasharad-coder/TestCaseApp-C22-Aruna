# QAGenie — Project State

> **Last updated:** 2026-05-08 (Session 1)
> **Always update this file at end of session.**

## App
- **Name:** QAGenie
- **Repo:** https://github.com/arunasharad-coder/TestCaseApp-C22-Aruna *(to be renamed)*
- **Live URL:** https://testcaseapp-c22-aruna.streamlit.app/ *(to be renamed in Phase 4)*
- **Tagline:** AI-powered test case generation for modern QA teams

## Current phase
**Phase 1 — User Controls (✅ COMPLETE)**

## Current task
Ready for Phase 2 — Selenium Java + TestNG output.

## What's working in development (v2 - not yet pushed)
- Configurable test count (3-10 via slider)
- Test type selection (Positive / Negative / Edge / Accessibility / Mixed)
- Authentication-aware mode (Path A — assumes session, generates auth-aware code)
- Dynamic button label matches selected count
- Existing v1 features still working (regression-tested)

## What's working in production (v1)
- Generates 5 Playwright TypeScript test cases from a feature description
- Multi-agent LangGraph workflow (Designer + Reviewer)
- Tavily search integration for real-time site context
- Jira CSV export
- Per-test-case Playwright code generation with download button
- Sidebar with tutorial video and prompt guidelines
- Guardrails for invalid/short input

## What's in progress
Phase 1 controls (started Session 1).

## What's next (in order)
1. **Phase 1 (current)** — Test count slider, test type dropdown, auth toggle
2. **Phase 2** — Selenium Java + TestNG output
3. **Phase 3** — Selenium Python + Playwright Python outputs
4. **Phase 4** — Migration to Hugging Face Spaces + README polish + screenshots
5. **Phase 5** — API testing mode (pytest+requests, REST Assured+TestNG)
6. **Phase 6** — Postman Collection export + sample prompts button
7. **Phase 7** — Requirement upload (PDF/docx → test cases)
8. **Phase 8** — Coverage matrix + test history (last 5 runs)

## Active decisions (locked in)
- **Hosting:** Streamlit Cloud now → migrate to HF Spaces in Phase 4 — *migration easier with stable code, not WIP*
- **Auth approach:** Path A (acknowledge limitation, generate auth-aware code with TODO comments) — *honest scope; production teams use storage state pattern anyway*
- **Framework priority:** Playwright TS (done) → Selenium Java → Selenium Python → Playwright Python — *Java first for biggest job-market signal*
- **Time budget:** 15 hrs/week
- **Cadence:** 1 LinkedIn post per shipped phase, every 3-5 days
- **Project goal:** Land a job in AI-augmented QA / SDET / QA Tools roles

## Known blockers / open questions
- None right now.

## Parking lot — NOT for v2
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

## Tech stack
- Python 3.14, Streamlit 1.57.0
- LangChain, LangGraph
- OpenAI GPT-4o-mini
- Tavily search
- python-dotenv

## Future direction (post-Phase 8) — known gaps and opportunities

QAGenie v2 generates test artifacts but does not yet address these deeper QA workflow problems. These are intentionally out of scope for v2 but represent the most valuable future work:

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

These three gaps represent v3 territory — substantial work, but the difference between QAGenie being a demo and being a production-grade SDET tool. Logged here so they're not forgotten.