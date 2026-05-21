# QAGenie — Project State

> **Last updated:** 2026-05-21 (Session 6 — Phase 3.5 in progress)
> **Always update this file at end of session.**

## App
- **Name:** QAGenie
- **Repo:** https://github.com/arunasharad-coder/TestCaseApp-C22-Aruna *(to be renamed in Phase 4)*
- **Live URL:** https://testcaseapp-c22-aruna.streamlit.app/ *(to be renamed in Phase 4)*
- **Tagline:** AI-powered test case generation for modern QA teams
- **Project goal:** Land a job in AI-augmented QA / SDET / QA Tools roles

## Current Status
**Phase 3.5 — UI polish (in progress, Session 6)**

- v2.4 (Phase 3.5) — UI polish in progress: B2 + C1/C2/C3 done locally, B1 in progress
- v2.3 (Phase 3) — Playwright Python chain added locally; not yet deployed
- v2.2 (Phase 2) — Selenium Java + TestNG, deployed to production Fri 5/15
- v2.1 — QAGenie branding (deployed earlier)
- v2.0 / v1 — Original Playwright TypeScript output, LangGraph workflow, Tavily search, Jira CSV export

## Recent Traction
- Post 3 (Phase 2 announcement, Mon 5/18) — 17K impressions, 87 likes as of Thu 5/21 morning
- LinkedIn About + Headline updated Wed 5/20 to position more confidently around AI-augmented QA tooling

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

## What's Committed Locally but NOT Deployed (v2.3 + v2.4 WIP)

**From v2.3 (Phase 3 — Playwright Python):**
- Playwright Python chain (senior-quality prompt with 10 requirements + DO NOT section)
- Framework dropdown extended to 3 options (PW TS, PW Python, Selenium Java + TestNG)
- Chain routing logic extended with `elif` for Playwright Python
- `code_language` logic converted from ternary to if/elif/else for 3-framework support
- Python syntax highlighting in code viewer

**From v2.4 (Phase 3.5 — in progress Thu 5/21):**
- ✅ B2 — Dynamic file extension for download button (`file_ext` derived from `framework`; label and file_name both now correct for .java / .py / .spec.ts)
- ✅ C1 — Sidebar "Automation" line generalized for multi-framework
- ✅ C2 — Sidebar tutorial "accurate Playwright scripts" → "accurate test scripts"
- ✅ C3 — Confirmed already neutral after C2 fix; no further change needed
- 🔄 B1 — Expander auto-opens when code exists (Approach C); ✅ added to TC header when code present (also closes L2)

## Immediate Next Session
**Phase 4 — Selenium Python** (planned for Fri 5/22)
- Add `selenium_python_chain` following the same senior-quality pattern as Java
- Add to framework dropdown (4th option)
- Update routing + `code_language` + `file_ext` mappings
- Deploy v2.4 + v2.5 together Fri 5/22
- Post 4 (Selenium Python + Phase 3.5 polish summary) scheduled Tue 5/26 (after Memorial Day weekend)

## Roadmap
1. **Phase 1** ✅ Posted — User controls (slider, type, auth toggle)
2. **Phase 2** ✅ Deployed Fri 5/15 — Selenium Java + TestNG (Post 3 = 17K impressions)
3. **Phase 3** ✅ Built locally — Playwright Python
4. **Phase 3.5** 🔄 In progress (Thu 5/21) — UI polish (B2, C1-C3 done; B1 in progress)
5. **Phase 4** — Selenium Python + HF Spaces migration + README polish — *functional + infra work; ships fast*
6. **Phase 5** — Polish Pass (UX upgrade) — *"Sweating the Streamlit details" — cards, indicators, empty states, fix B1 tab-reset quirk, L1 horizontal scroll workaround, README polish. Framed as a content moment: prompt-engineering for UX. See Post 5 idea below.*
7. **Phase 6** — API testing mode (pytest+requests, REST Assured+TestNG)
8. **Phase 7** — Postman Collection export + sample prompts button
9. **Phase 8** — Requirement upload (PDF/docx → test cases)
10. **Phase 9** — Coverage matrix + test history (last 5 runs)

### Post ideas tied to roadmap
- **Post 4 (Tue 5/26):** Phase 3.5 polish + Selenium Python launch — *"Small UI fixes, big UX wins"*
- **Post 5 (post-Phase 5):** Polish Pass deep dive — *"Sweating the details: turning Streamlit quirks into UX wins (with the prompts I used)"* — Aruna's idea from Session 6. Process post, not announcement.

## Locked Decisions
- **Hosting:** Streamlit Cloud now → migrate to HF Spaces in Phase 4 — *migration easier with stable code, not WIP*
- **Auth approach:** Path A (acknowledge limitation, generate auth-aware code with TODO comments) — *honest scope; production teams use storage state pattern anyway*
- **Framework rollout order:** Playwright TS (done) → Selenium Java (done) → Playwright Python (done locally) → Selenium Python (Phase 4) — *Java first for biggest job-market signal*
- **UI framework:** Stay with Streamlit through Phase 9 (decided Fri 5/15 — see UI Framework Decision below)
- **Time budget:** 15 hrs/week
- **Post cadence:** 1 LinkedIn post per shipped phase, every 3-5 days
- **Deploy cadence:** Coordinate deploys with LinkedIn posts — never deploy a major feature without an accompanying announcement

## Known UI Issues

### Phase 3.5 — Done (Thu 5/21)

- ✅ **B2:** Download button label + file_name now dynamic via `file_ext` derived from `framework`
- ✅ **C1:** Sidebar Automation line generalized
- ✅ **C2:** "accurate Playwright scripts" → "accurate test scripts"
- ✅ **C3:** Confirmed already neutral after C2 fix
- 🔄 **B1 (partial fix):** Approach C — expander auto-opens when code exists for that test case, ✅ shown on header. **Trade-off accepted:** tab still resets to "Manual Steps" on rerun (Streamlit limitation, no native API to control tab state). Full fix deferred to Phase 5 polish pass.
- ✅ **L2:** TC headers now show ✅ when code generated (rolled into B1 fix)

### Phase 5 polish-pass backlog

- **B1 tab-reset (deferred):** Restructure display loop — replace `st.tabs()` with controllable widget pattern (radio + container, or render-outside-tabs) so generated code is visible without tab switch. Estimated 45-60 min as part of broader polish phase.
- **L1:** Code block scrolls horizontally on long lines — Streamlit limitation, explore wrap workarounds or scroll affordance
- **UX upgrade:** Cards for test cases, generation-state indicators, empty states ("no test cases yet" placeholder)
- **B1 expander auto-reopen quirk:** Currently if user manually closes an expander that has code, it re-opens on next rerun. Minor — accept for now, revisit during polish pass if it surfaces as user pain.

## UI Framework Decision

**Decision:** Stay with Streamlit through Phase 9.

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

## Future Direction / V3 Gaps (post-Phase 9)

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