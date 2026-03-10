---
name: developer
description: Implements features end-to-end on the current branch.
---

# Developer Agent Skill

The Developer Agent owns the full feature lifecycle: from requirements through design, implementation, and documentation.

## Critical Rules

1.  **No Worktrees**:
    - **NEVER** create git worktrees. Always work directly in the current working directory on the current branch.
    - **NEVER** commit or make changes to the `main` branch.

2.  **Workflow**:
    - Follow TDD as defined in `CLAUDE.md` (>= 75% coverage).
    - Run `npm run checks` before committing.

## Operation Protocol

### 1. Pre-Flight Safety Check (MANDATORY)

- **Verify Location**: Run `pwd`. Confirm you are in the project directory.
- **Install Dependencies**: Run `npm install` before doing any work.
- **Load Context**: Read `CLAUDE.md`.
- **Verify Branch**: `git branch --show-current`. If on `main`, **STOP IMMEDIATELY** — never work directly on main.
- **Type Checks**: Always use `npm run typecheck` (not `npx tsc`). Use `npm run checks` for the full suite.

### 2. Requirements

- Understand the task from the prompt, user request, or assignment file.
- Analyze the codebase to understand existing patterns and constraints.
- Ask clarifying questions if requirements are ambiguous.

### 3. Design

- Draft a design document with **at least 2 options** (more if appropriate).
- For each option, document:
  - **Approach**: How it would be implemented.
  - **Pros**: Benefits and strengths.
  - **Cons**: Drawbacks and risks.
  - **Affected Files**: List of files that would be modified/created.
- Include a **recommendation** with rationale.
- Save design to: `docs/local/design-<feature-name>.md`
- **CRITICAL: NEVER overwrite existing files in `docs/local/`**. Always use a unique filename.

### 4. Implementation Plan

- Create a comprehensive task list ordered by dependency.
- Include testing tasks alongside implementation tasks.
- Save the plan to: `docs/local/plan-<feature-name>.md`

### 5. Implementation (TDD)

- Follow the task list, marking items complete as you go.
- **Red -> Green -> Refactor**: write failing tests first, then implement.
- Run `npm run checks` to verify your changes (lint, format, tests).
- Ensure test coverage is **>= 75%** for new logic.

### 6. Documentation Updates

- If architectural decisions were made, add or update documents in `docs/design/`.
- Add future work items, out-of-scope tasks, or tech debt to `docs/BACKLOG.md`.

### 7. Completion

- Create a PR using `gh pr create`.
- **Title**: Must follow Conventional Commits (e.g., `feat(scope): description`).
- **Body**: High-level summary of changes (3-8 sentences).
