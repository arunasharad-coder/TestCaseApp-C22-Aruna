# QAGenie Changelog

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