# QAGenie — Project State

> **Last updated:** 2026-05-21 (Session 6 — Phase 3.5 + Phase 3 + Phase 4a SHIPPED to prod)
> **Always update this file at end of session.**

## App
- **Name:** QAGenie
- **Repo:** https://github.com/arunasharad-coder/TestCaseApp-C22-Aruna *(to be renamed in Phase 4b)*
- **Live URL:** https://testcaseapp-c22-aruna.streamlit.app/ *(to be renamed in Phase 4b after HF Spaces migration)*
- **Tagline:** AI-powered test case generation for modern QA teams
- **Project goal:** Land a job in AI-augmented QA / SDET / QA Tools roles

## Current Status

**Top of mind (as of 2026-05-27 EOD):**
- Phase 4b queued — HF Spaces migration + README polish (~2-2.5 hr, no time pressure, no LinkedIn post gated on it)
- Post 4 live since 2026-05-27 10:21 AM CT — 5 commenters, mixed engagement quality (1 substantive, 2 warm, 1 vague-critical, 1 architect-defender)
- 3 LinkedIn replies drafted, awaiting post (Nikhil, Theresa, Keber)
- Deliberate non-replies to Maksim and Cristian
- Cowork adoption deferred to post-Phase 4b (Phase 5 timeframe at earliest)

**Phase 4a — Selenium Python SHIPPED to production (Thu 5/21 afternoon)**

Single deploy on Thu 5/21 evening bundled three phases of work:
- v2.5 (Phase 4a) — Selenium Python chain added; 4-framework support live in prod
- v2.4 (Phase 3.5) — Dynamic file extension (B2) + sidebar copy (C1/C2/C3) live in prod
- v2.3 (Phase 3) — Playwright Python chain live in prod

Live URL now generates code in 4 frameworks: Playwright TS, Playwright Python, Selenium Java + TestNG, Selenium Python.

- v2.2 (Phase 2) — Selenium Java + TestNG, deployed Fri 5/15
- v2.1 — QAGenie branding (deployed earlier)
- v2.0 / v1 — Original Playwright TypeScript output, LangGraph workflow, Tavily search, Jira CSV export

## Recent Traction
- Post 3 (Phase 2 announcement, Mon 5/18) — ~20K impressions, 87+ likes, 1 substantive comment as of Thu 5/21 evening
- First architect-level comment received: senior Cloud/Tech Architect (18+ years) validating the engineering approach and pointing toward the framework-integration gap (POM, framework config, CI/CD) as the real opportunity. Shaped V3 roadmap thinking — see Future Direction section.
- LinkedIn About + Headline updated Wed 5/20 to position more confidently around AI-augmented QA tooling
- Post 4 drafted Thu 5/21 evening; scheduled for Tue 5/26 9-11 AM CT (Coverage-led + honest gap framing)

## What's Deployed in Production
- Configurable test count (3-10 via slider)
- Test type selection (Positive / Negative / Edge / Accessibility / Mixed)
- Authentication-aware mode (Path A — assumes session)
- Dynamic button label matches selected count
- QAGenie branding (renamed from "QA Test Cases Generator")
- Tagline visible below title
- **Framework selector dropdown — 4 options:** Playwright TypeScript, Playwright Python, Selenium Java + TestNG, Selenium Python
- `selenium_java_chain` with senior-level prompt (TestNG annotations, WebDriverWait, groups, priority, Javadoc)
- `playwright_python_chain` with senior-level prompt (pytest + sync_playwright + semantic locators + markers)
- `selenium_python_chain` with senior-level prompt (pytest + `@pytest.fixture` for driver + `webdriver-manager` + function-based + docstrings)
- Dynamic button label matches selected framework
- Dynamic tab label matches selected framework
- **Dynamic file extension on download button** (`.spec.ts` / `.py` / `.java`) derived from framework
- Syntax highlighting matches framework (typescript / python / java)
- Sidebar copy generalized across frameworks (no more "Playwright-only" language)
- v1 features: 5 test cases default, LangGraph workflow, Tavily search, Jira CSV export, per-TC code generation, sidebar with tutorial

## Immediate Next Session
**Phase 4b — HF Spaces migration + README polish** (when bandwidth allows next week — no time pressure since no LinkedIn post is gated on this)
- HF Spaces migration: secrets management, requirements.txt Python compatibility check, DNS/URL update, LinkedIn link refresh
- README polish: header + badges + screenshot, quick start (clone/install/env/run), feature list, tech stack, architecture diagram (optional), roadmap, known limitations
- Estimated: 2-2.5 hours total
- No urgent timeline — Streamlit Cloud deploy is live and stable; HF migration is infra hygiene, not user-facing

## Roadmap
1. **Phase 1** ✅ Posted — User controls (slider, type, auth toggle)
2. **Phase 2** ✅ Deployed Fri 5/15 — Selenium Java + TestNG (Post 3 = ~20K impressions)
3. **Phase 3** ✅ Shipped Thu 5/21 — Playwright Python
4. **Phase 3.5** ✅ Shipped Thu 5/21 — UI polish (B2, C1-C3 shipped; B1 deferred to Phase 5)
5. **Phase 4a** ✅ Shipped Thu 5/21 — Selenium Python (Post 4 scheduled Tue 5/26)
6. 6. **Phase 4b** — HF Spaces deployment + dependency cleanup — ✅ **DONE 2026-05-29**
   - HF Space live: https://huggingface.co/spaces/arunasharad/qagenie
   - GitHub Actions auto-sync workflow active (`.github/workflows/sync-to-hf.yml`)
   - `requirements.txt` cleaned (`langchain` umbrella removed; 9 deps remaining, all verified active)
   - Dockerfile added for HF Spaces (Docker SDK + Streamlit template)
   - HF Space secrets configured: `OPENAI_API_KEY`, `TAVILY_API_KEY`
   - Both production targets (Streamlit Cloud + HF Spaces) deploy from same GitHub `main` branch
   - **Remaining for Phase 4b that was descoped:** Full README polish (recruiter-facing portfolio version). Current README has minimal body + YAML frontmatter — functional but not polished. Bundled into Phase 5 since the polish work is UX/presentation, which fits Phase 5's "sweating the details" theme better than 4b's infra theme.
7. **Phase 5** — Polish Pass (UX upgrade)
   - **Headline fix: B1 tab-reset bug** via display-loop restructure 
     (see B1 details below). Dissolves L2 as byproduct.
   - "Sweating the Streamlit details" — cards, indicators, empty states
   - L1 horizontal scroll workaround
   - Rewrite Playwright TS prompt to senior-quality (currently lags the 
     other 3 chains)
   - Refresh Selenium Java prompt for WebDriverManager + Javadoc placement
   - Refactor `code_language`/`file_ext` into single FRAMEWORK_CONFIG dict
   - README polish (if not already done in 4b)
   - **Framing for Post 6:** prompt-engineering for UX
**B1 details (verified 2026-05-29):**
- **Symptom:** After clicking "Generate Code for TC N" inside a test case 
  expander, the tab indicator snaps back to "Manual Steps" even though the 
  generated code is correctly rendered below. Visual mismatch: code visible, 
  wrong tab highlighted.
- **Confirmed in production** via testcaseapp-c22-aruna.streamlit.app on 
  2026-05-29 — not a local-only artifact.
- **Root cause:** Streamlit's script-rerun model. Clicking Generate Code 
  triggers a full script rerun; `st.tabs()` has no session-state binding, 
  so it always re-initializes with index 0 (first tab = Manual Steps).
- **Why prior attempts stalled (Phase 3.5):** "Approach C" tried to work 
  *around* `st.tabs()` limitations with nested `with` blocks and hit 
  indentation issues. The lesson: don't fight `st.tabs()`, replace it.
- **Recommended fix:** Swap `st.tabs(["📝 Manual Steps", script_tab_label])` 
  for `st.radio()` styled as tabs, with the selected value bound to 
  `st.session_state[f"active_tab_{i}"]`. Streamlit's radio widget *does* 
  persist via session_state across reruns. This is also why Phase 5 frames 
  this as a display-loop restructure, not a patch.
- **Bundled fix opportunity:** Same restructure dissolves L2 (the ✅ marker 
  on TC headers after code generation) — both are blocked by the same 
  st.tabs() limitation.
- **Ruled out:** `langchain` umbrella package removal (tested 2026-05-29) — 
  bug exists identically in production with langchain installed AND in 
  cleaned local venv without it. Cleanup is not the cause.
8. **Phase 6** — API testing mode (pytest+requests, REST Assured+TestNG)
9. **Phase 7** — Postman Collection export + sample prompts button
10. **Phase 8** — Requirement upload (PDF/docx → test cases)
11. **Phase 9** — Coverage matrix + test history (last 5 runs)

### V3 Roadmap (Framework Integration — the architect-comment direction)
Surfaced from Post 3 commenter (senior architect, 18+ years) Thu 5/21. This is the "maintainable, framework-aware test generation" direction. ~6 months of credible portfolio work, each phase post-worthy:

12. **Phase 10** — Page Object Model generation (generate Page class + Test class as 2 files per test instead of one)
13. **Phase 11** — Framework config generation (`pytest.ini`, `testng.xml`, `playwright.config.ts` as separate downloadables)
14. **Phase 12** — CI/CD scaffolding (GitHub Actions YAML for each framework)
15. **Phase 13+** — Codebase awareness (upload existing test framework, AI matches team conventions). True V3 territory; multi-step agent + longer context windows + file parsing. Multi-month engineering investment.

### Post ideas tied to roadmap
- **Post 4 (Tue 5/26):** 4 frameworks + Phase 3.5 polish — coverage-led + honest gap (drafted, scheduled)
- **Post 5 (post-Phase 5):** Polish Pass deep dive — *"Sweating the details: turning Streamlit quirks into UX wins (with the prompts I used)"*. Process post, not announcement. Potential video format (Post 2 was video, 3 and 4 are screenshots — variety is good for algorithm).
- **Post 6 (post-Phase 10):** Page Object Model generation — first concrete step toward framework-aware AI test generation
- **Post 7+ (Phases 11-13):** One per shipped phase along the V3 framework-integration arc

## Locked Decisions
- **Hosting:** Streamlit Cloud now → migrate to HF Spaces in Phase 4b — *migration easier with stable code (4-framework prod confirmed Thu 5/21)*
- **Auth approach:** Path A (acknowledge limitation, generate auth-aware code with TODO comments) — *honest scope; production teams use storage state pattern anyway*
- **Framework rollout order:** Playwright TS (done) → Selenium Java (done Phase 2) → Playwright Python (done Phase 3) → Selenium Python (done Phase 4a) — *Java first for biggest job-market signal; all 4 frameworks shipped Thu 5/21*
- **UI framework:** Stay with Streamlit through Phase 9 (decided Fri 5/15 — see UI Framework Decision below)
- **Time budget:** 15 hrs/week
- **Post cadence:** 1 LinkedIn post per shipped phase, every 3-5 days
- **Deploy cadence:** Coordinate deploys with LinkedIn posts — never deploy a major feature without an accompanying announcement

**Engagement intelligence — Post 4 (2026-05-27)**
- Substantive commenters worth long-term relationship: Keber Flores (`@keber/qa-framework`), Cristian N. (AI-Driven QA Architect)
- Warm cheerleaders: Theresa DiStefano, Nikhil Guzar
- Vague/critical: Maksim Maksimov — non-reply decision
- Pattern observed: Post 4's "v3 is headed toward maintainable, framework-aware tests" paragraph drew the substantive commenters. Honest-gap framing + crediting prior commenter + naming forward direction = the architecture that invites thoughtful engagement.
- Action: continue this framing pattern in Posts 5+.

**Engagement quality validating direction (Post 3 → Post 4):**
- Post 3 (Phase 2): first architect-level engagement (Cloud/Tech Architect, 18+ years) — pointed toward framework integration (POM, config, CI/CD) as the real opportunity. Reshaped V3 roadmap (Phases 10-13).
- Post 4 (Phase 3 + 3.5 + 4a): second architect-level engagement (Cristian N., AI-Driven QA Architect) + Keber Flores sharing a complementary methodology framework (`@keber/qa-framework`). Confirms direction: AI-augmented QA at the framework/maintainability layer is where senior QA people are paying attention.

## Known UI Issues

### Phase 3.5 — Shipped (Thu 5/21)

- ✅ **B2:** Download button label + file_name now dynamic via `file_ext` derived from `framework`
- ✅ **C1:** Sidebar Automation line generalized
- ✅ **C2:** "accurate Playwright scripts" → "accurate test scripts"
- ✅ **C3:** Confirmed already neutral after C2 fix

### Phase 5 polish-pass backlog

**UI / UX**
- **B1 (deferred from Phase 3.5):** Expander/tab state resets after Generate Code click. Approach C attempted Session 6 but stalled — nested `with` indentation compounded across edits and Streamlit's tab-state limitation means even a clean fix only solves half the problem. Phase 5 will restructure the display loop entirely (replace `st.tabs()` with a controllable widget pattern), which dissolves B1 as a byproduct. Estimated 45-60 min as part of the broader polish phase.
- **L1:** Code block scrolls horizontally on long lines — Streamlit limitation, explore wrap workarounds or scroll affordance
- **L2:** TC headers don't visually indicate if code has been generated (✅ on header) — was bundled with B1 attempt, also deferred. Trivial to add once the display loop is restructured.
- **UX upgrade:** Cards for test cases, generation-state indicators, empty states ("no test cases yet" placeholder)
- **Framework-specific reporting tips banner:** When framework changes, show small info hint with the right report command (e.g. PW TS → `npx playwright test --reporter=html`; PW Python / Selenium Python → `pytest --html=report.html` + note `pip install pytest-html`; Selenium Java → "View reports in `test-output/index.html`"). Adds production-grade hygiene without modifying generated test files.

**Prompt quality (revealed by 4-framework production comparison Thu 5/21)**
- **Playwright TypeScript prompt rewrite (HIGH priority):** Current prompt is v1 — just 3 lines, no requirements list, no DO NOT section. Result: output includes markdown fences, explanatory text before/after code, and at least one invalid Playwright assertion (`toHaveCountGreaterThan` doesn't exist). It's the default framework option AND the lowest-quality output. Senior-quality rewrite using the same 10-requirement + DO NOT template as the other 3 chains. ~15 min focused work. Also great narrative angle for Post 5 ("My first prompt was 3 lines; here's what changed").
- **Selenium Java prompt refresh (MEDIUM):** Currently uses manual driver path (`System.setProperty("webdriver.chrome.driver", ...)`). Modern standard is `WebDriverManager.chromedriver().setup();` (Boni García's library). Also generated output places Javadoc between `@Test` annotation and method signature — should be above the annotation per Java convention. Small prompt tweaks.

**Code structure**
- **Refactor `code_language` + `file_ext` into single `FRAMEWORK_CONFIG` dict:** Currently two parallel if/elif/else blocks derived from same `framework` value. Single source of truth would be cleaner:
  ```python
  FRAMEWORK_CONFIG = {
      "Playwright TypeScript":  {"language": "typescript", "ext": "spec.ts", "chain": playwright_chain},
      "Playwright Python":      {"language": "python",     "ext": "py",      "chain": playwright_python_chain},
      "Selenium Java + TestNG": {"language": "java",       "ext": "java",    "chain": selenium_java_chain},
      "Selenium Python":        {"language": "python",     "ext": "py",      "chain": selenium_python_chain},
  }
  ```
  Also collapses the chain routing if/elif into `chain = FRAMEWORK_CONFIG[framework]["chain"]`. Eliminates 3 parallel if/elif blocks → 1 dict lookup. Easy fix once we're in restructure mode.

**Dependency hygiene**
- **LangChain deprecation warnings:** Two warnings flagged in terminal Thu 5/21:
  1. `LangChainPendingDeprecationWarning: The default value of allowed_objects will change...` (langgraph internal)
  2. `LangChainDeprecationWarning: TavilySearchResults was deprecated in LangChain 0.3.25` — should switch to `langchain-tavily` package and import as `from langchain_tavily import TavilySearch`
  Both still work. Address during Phase 5 or a dedicated dependency-update session.

## Session 6 Notes (Thu 5/21) — Lessons from B1 attempt

Reflections worth preserving while fresh:

1. **The pattern fix worked; the integration didn't.** The Approach C logic (`has_code = f"pw_code_{i}" in st.session_state` → `expanded=has_code`) was correct in isolation. What broke us was integrating it into a nested-`with` block while making other edits in the same pass. Each indentation fix introduced a new misalignment somewhere else.

2. **Streamlit's `with` blocks are fragile to edit.** A `for` loop containing `with st.expander()` containing `st.tabs()` containing `with tab_manual:` containing `st.columns()` containing `with col1:` is 5 levels of nesting. Single-line edits within this structure are high-risk. Phase 5 should flatten this — either by extracting render helpers (`render_manual_tab(tc)`, `render_script_tab(tc, framework, i)`) or by replacing tabs with a simpler pattern.

3. **Copilot fought the revert.** When undoing the B1 attempt, Copilot kept suggesting the `has_code` lines as ghost completions because its context memory was still on the WIP direction. Lesson: during revert work, dismiss AI suggestions aggressively or temporarily disable Copilot.

4. **The 2-hour budget was the right scope; B1 was the wrong scope.** B2 + C1/C2/C3 took ~30 min total and landed cleanly. B1 ate the remaining 90 min and didn't ship. Future scoping principle: **prefer multiple small bug fixes over one medium-complexity refactor in the same session.** If a fix needs restructuring, it belongs in a dedicated polish phase, not a fix phase.

5. **Phase 5 framing crystallized today.** Aruna's idea — make the UX polish its own "prompt engineering for UX" content moment — is now the explicit framing for Phase 5. B1, L1, L2, plus the UX upgrade work all belong together. Treating them as one cohesive phase (instead of bolted-on tasks) gives Post 5 a strong narrative.

### Afternoon — Phase 4a Selenium Python ship

6. **The prompt template is now mature.** Selenium Python chain came together in ~30 min of focused work after lunch. The 10-requirements + DO NOT template, refined through Selenium Java and Playwright Python, just *worked*. First-shot output from the LLM matched all 10 requirements without iteration. This template is now a reusable asset for future framework chains (REST Assured, Cypress, etc. if those ever happen).

7. **Production validation surfaced new findings.** Side-by-side comparison of all 4 frameworks' actual prod output (first time seen together) revealed:
   - Playwright TS prompt is the weakest (v1, predates the senior-quality template) — output includes markdown fences, explanatory text, and an invalid assertion. **Now Phase 5 backlog as HIGH priority.**
   - Selenium Java prompt has minor staleness: manual driver path instead of WebDriverManager, Javadoc placement quirk. **Now Phase 5 backlog as MEDIUM.**
   - Playwright Python silently substituted real Google selectors (`input[name='q']`) instead of using the provided made-up ones (`#searchbar`). The other 3 chains used the provided selectors faithfully. This is the **selector accuracy / Gap 2** problem manifesting in practice — not a bug, but inconsistency. Interesting talking point for V3 work.

8. **Architect-level engagement on Post 3 reshaped V3 thinking.** A senior Cloud/Tech Architect (18+ years) commented validating the engineering approach AND pointing toward framework-integration (POM, framework config, CI/CD) as "the real opportunity." This isn't a generic compliment — it's a market signal from someone in the buyer profile. V3 roadmap (Phases 10-13) added to capture this direction. The comment also shaped Post 4's "v3 is headed" close.

9. **Three phases in one deploy is a real velocity moment.** Phase 3.5 + Phase 3 (Playwright Python that had been sitting locally for ~a week) + Phase 4a all shipped together Thu 5/21 evening. The single push deployed: B2/C1/C2/C3 polish fixes + 2 new frameworks. Sometimes the right move is to batch — would have been worse to push three separate times for trickle deploys, each begging for its own LinkedIn post.

10. **Lunch break saved the day.** Morning B1 session was getting tangled. Stopping cleanly + lunch break + restarting fresh on a different scope (Phase 4a) was the productivity unlock. Naming this so future-Aruna remembers: **when stuck, change task or take a break. Don't push through.**

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
- BDD/Gherkin output
- Multi-language UI
- Critic agent / scoring
- Slack/email notifications

*Note: "Page Object Model generation" was previously in this parking lot; promoted to V3 roadmap (Phase 10) Thu 5/21 after Post 3 architect comment validated it as a real opportunity.*

## Tech Stack
- Python 3.14, Streamlit 1.57.0
- LangChain, LangGraph
- OpenAI GPT-4o-mini (temperature=0)
- Tavily search (max_results=2)
- python-dotenv