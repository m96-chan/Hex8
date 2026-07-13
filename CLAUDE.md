# CLAUDE.md

Project rules for working on Hex8. These are binding constraints, not suggestions.

## 1. No Ticket, No Do

Every change must be tied to a GitHub Issue before any code is written. If no issue covers the work, create one (or ask the user to) before starting. Do not start implementation "just to see" without a ticket.

## 2. Plan first

Always produce a plan before implementing. Use plan mode (or an explicit written plan) for any non-trivial task, and get it aligned with the user before writing code.

## 3. Test-Driven Development is mandatory

- Write the failing test before the implementation, for every change.
- Target test coverage of 90% or higher per module as a recommended bar.
- No implementation commit without accompanying tests.

## 4. Documentation updates: before AND after

- **Before** starting work on an issue: update the relevant documentation (README, docstrings, design docs, the issue itself) to reflect the intended design, so the plan is written down before code changes it.
- **After** completing work: update the documentation again to reflect what was actually built, including any deviation from the original plan.

## 5. Never guess specifications

If a spec, format, threshold, or behavior is ambiguous or not explicitly defined (in the README, an issue, or prior discussion), do not assume or infer it. Stop and ask the user. This applies to numeric parameters, file formats, edge-case behavior, naming, etc. — anything not explicitly stated is unknown, not "probably X".

## 6. GitHub Issues are external memory

- Treat GitHub Issues as the durable record of project state, not local notes or memory.
- Append progress, decisions, and findings directly to the relevant Issue as comments, continuously as work proceeds — not just at the end.
- All Issue content (titles, bodies, comments) must be written in **English**.

## 7. Language policy

- All documentation and source code (comments, docstrings, commit messages, issue text) must be written in **English**.
- Conversational replies to the user (chat responses, questions, confirmations) must be in **Japanese**.

## 8. No stubs or mocks in real code

- Stub/mock implementations must not be used in production code paths.
- Exception: a stub may exist temporarily for a prerequisite/leading task that a later task depends on — but this should be avoided where possible.
- If creating a stub in this sequencing sense seems necessary, do not create it silently — ask the user for confirmation first.
