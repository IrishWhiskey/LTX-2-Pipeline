---
name: code-reviewer
description: Reviews all changes in a feature branch. Does not modify code.
---

# Code Reviewer Skill

Performs a comprehensive code review of all changes in the current feature branch. Reports findings but does NOT make code changes.

## Workflow

### 1. Determine Base Branch

```bash
# Auto-detect main or master:
git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@'
# Fallback: check if 'main' exists, then 'master'
```

### 2. Gather Changes

```bash
# Get the merge base (common ancestor)
MERGE_BASE=$(git merge-base <base-branch> HEAD)

# List all changed files
git diff --name-only $MERGE_BASE HEAD

# Get the full diff for analysis
git diff $MERGE_BASE HEAD
```

### 3. Analyze Each Changed File

For each modified file:

1. Read the current file content
2. Understand the context of changes
3. Evaluate against review criteria (see below)

### 4. Decision Point

**If code is great (no issues found):**

- Output `APPROVED` and exit
- No report file is created

**If issues are found:**

- Save report to `docs/local/review-X.md` (where X is incremented for each review iteration)
- Output `NEEDS FIXES` with a summary

---

## Review Criteria

Evaluate changes against these categories:

### Critical Issues

Issues that **must** be fixed before merging:

- Security vulnerabilities (SQL injection, XSS, command injection, hardcoded secrets)
- Data integrity bugs (race conditions, data loss, incorrect state mutations)
- Breaking changes (unintended API contract violations)

### Bugs & Logic Errors

- Off-by-one errors, null/undefined handling, incorrect conditionals
- Missing error handling for likely failure cases
- Incorrect async/await usage, unhandled promise rejections
- Edge cases that will cause crashes or incorrect results

### Performance Concerns

- N+1 queries, missing database indexes
- Unnecessary re-renders, missing memoization
- Memory leaks, unbounded data structures
- Inefficient algorithms for the data scale

### Code Quality

- Overly complex functions that should be broken down
- Duplicated logic that should be abstracted
- Poor naming that obscures intent

### Refactoring Opportunities

Actively look for code that should be refactored:

- Functions exceeding ~30 lines
- Deep nesting (3+ levels)
- Duplicated logic (3+ occurrences)
- Boolean function parameters
- Long parameter lists (4+)
- Comments explaining "what" code does
- Mixed abstraction levels

### Style & Conventions

- Deviations from project patterns and conventions

---

## Report Format

Save to: `docs/local/review-X.md`

Where `X` starts at 1 and increments for each review iteration.

```markdown
# Code Review: <branch-name>

**Date**: <timestamp>
**Base Branch**: <base-branch>
**Files Changed**: X files (+Y/-Z lines)

## Summary

[2-3 sentence overview of the changes and issues found]

## Critical Issues

[List items or "None"]

## Bugs & Logic Errors

[List items or "None"]

## Performance Concerns

[List items or "None"]

## Code Quality

[List items or "None"]

## Refactoring Opportunities

[List items or "None"]

## Style & Conventions

[List items or "None"]

---

**Status**: NEEDS FIXES
```

### Issue Format

Each issue should include:

- **File and line**: `src/components/Auth.tsx:45-52`
- **Description**: Clear explanation of the problem

Example:

```markdown
### `src/api/users.ts:23-28`

**SQL Injection vulnerability**

User input is interpolated directly into the query without parameterization.
```

---

## Output

**When APPROVED:**

```
APPROVED

No issues found. Code is ready for PR.
```

**When issues found:**

```
NEEDS FIXES

Found X issues. Report saved to: docs/local/review-X.md
```
