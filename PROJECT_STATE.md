# QAGenie — Project State

> **Last updated:** 2026-05-11 (Session 2)
> **Always update this file at end of session.**

## App
- **Name:** QAGenie
- **Repo:** https://github.com/arunasharad-coder/TestCaseApp-C22-Aruna *(to be renamed)*
- **Live URL:** https://testcaseapp-c22-aruna.streamlit.app/ *(to be renamed in Phase 4)*
- **Tagline:** AI-powered test case generation for modern QA teams

## Current phase
**Phase 2 — Selenium Java + TestNG output (✅ COMPLETE — code committed locally)**

## Current task
Phase 2 committed locally. Awaiting strategic deploy window (Wed/Thu) before pushing to production and publishing LinkedIn Post 3.

## What's deployed in production (v2.1)
- Configurable test count (3-10 via slider)
- Test type selection (Positive / Negative / Edge / Accessibility / Mixed)
- Authentication-aware mode (Path A — assumes session)
- Dynamic button label matches selected count
- QAGenie branding (renamed from "QA Test Cases Generator")
- Tagline visible below title
- v1 features: 5 test cases default, LangGraph workflow, Tavily search, Jira CSV export, per-TC Playwright code generation, sidebar with tutorial

## What's committed locally but NOT YET deployed (v2.2)
- Framework selector dropdown (Playwright TypeScript vs Selenium Java + TestNG)
- selenium_java_chain with senior-level prompt (TestNG annotations, WebDriverWait, groups, priority, Javadoc)
- Dynamic button label matches selected framework
- Dynamic tab label matches selected framework
- Java syntax highlighting in code viewer
- Regression-tested: Playwright TypeScript output still works

## What's in progress
Nothing in progress — Phase 2 fully complete locally.

## What's next (in order)
1. **Wed/Thu deploy** — push Phase 2 to production + publish LinkedIn Post 3
2. **Phase 3** — Selenium Python + Playwright Python outputs
3. **Phase 4** — Migration to Hugging Face Spaces + README polish + screenshots
4. **Phase 5** — API testing mode (pytest+requests, REST Assured+TestNG)
5. **Phase 6** — Postman Collection export + sample prompts button
6. **Phase 7** — Requirement upload (PDF/docx → test cases)
7. **Phase 8** — Coverage matrix + test history (last 5 runs)

## ⏰ Pending deploys
- **Phase 2 (Selenium Java + TestNG)** — committed locally on 2026-05-11. Deploy scheduled for Wed/Thu 2026-05-13/14 alongside LinkedIn Post 3. Reason for delay: Post 1 (Phase 1) published 2026-05-11; need 48-72hr engagement window before next post.

### To deploy on Wed/Thu:
1. `cd ~/qagenie && source .venv/bin/activate`
2. `git push` (deploys 2 commits — Phase 2 code + tracking updates)
3. Wait ~2 min for Streamlit Cloud to redeploy
4. Verify production at https://testcaseapp-c22-aruna.streamlit.app/ shows framework dropdown
5. Test both frameworks generate code on production
6. Publish LinkedIn Post 3 (draft already in LINKEDIN_POSTS.md)

## Active decisions (locked in)
- **Hosting:** Streamlit Cloud now → migrate to HF Spaces in Phase 4 — *migration easier with stable code, not WIP*
- **Auth approach:** Path A (acknowledge limitation, generate auth-aware code with TODO comments) — *honest scope; production teams use storage state pattern anyway*
- **Framework priority:** Playwright TS (done) → Selenium Java (done locally) → Selenium Python → Playwright Python — *Java first for biggest job-market signal*
- **Time budget:** 15 hrs/week
- **Cadence:** 1 LinkedIn post per shipped phase, every 3-5 days
- **Deploy cadence:** Coordinate deploys with LinkedIn posts — never deploy a major feature without an accompanying announcement
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