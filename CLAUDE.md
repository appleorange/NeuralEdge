# NeuralEdge — Claude Code Instructions

## Project Overview
AI/ML-based stock trading bot using sentiment analysis + technical indicators.
Stack: Python, Alpaca API, FinBERT, XGBoost, SQLite, Streamlit.
Tools: gsd and gstack for task/stack management.

## Session Start Checklist
- Read docs/ROADMAP.md before doing anything
- Check which tasks are in progress or next up
- Review recent lessons in docs/lessons.md if it exists
- Confirm current git branch before making changes

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update docs/lessons.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer be proud of this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it, don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## NeuralEdge-Specific Rules
- NEVER hardcode API keys — always use .env
- ALWAYS default to paper trading mode unless --live flag is explicitly passed
- Every trade must be logged to SQLite with full context
- Risk manager must be consulted before every order execution
- When modifying the ML model, always run backtest before marking done
- After each completed module, update docs/ROADMAP.md accordingly

## MCP Servers
- Use context7 MCP for up-to-date library docs before implementing any
  library feature (alpaca-py, XGBoost, FinBERT, APScheduler, Streamlit)
- When implementing any library, fetch its current docs via context7 first

## Documentation Maintenance
After completing any module or phase, update ALL of these:
- docs/changelog.md — date + what was built
- docs/project_status.md — current phase + what's next
- docs/architecture.md — if any data flow changed
- docs/ROADMAP.md — mark completed tasks [x]

## Git Discipline
- Commit after every completed, working module
- Commit messages: "feat: ", "fix: ", "refactor: " prefixes
- Never commit broken code to main
