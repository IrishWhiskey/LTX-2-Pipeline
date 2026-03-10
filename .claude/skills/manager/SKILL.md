---
name: manager
description: Manages worktrees and merges PRs for multi-agent development.
---

# Manager Agent Skill

The Manager Agent creates isolated worktrees for feature development and merges completed work back to main.

## Responsibilities

1. **Worktree Management**: Create and manage git worktrees for feature branches.
2. **Merge Protocol**: Squash-merge completed PRs following project conventions.

## Worktree Creation

1. **Check for Branch**:

   ```bash
   git branch --list feat/<feature-name>
   ```

   - If it exists: `git worktree add ../<feature-name> feat/<feature-name>`
   - If new: `git worktree add ../<feature-name> -b feat/<feature-name>`

2. **Infrastructure Copy (CRITICAL)**:

   ```bash
   cp .env.local ../<feature-name>/.env.local
   ```

## Merge Protocol

### Squash & Merge

- **Method**: Squash and Merge via `gh pr merge --squash`.
- **Title**: Conventional Commits format (e.g., `feat(scope): description`).
- **Body**: 3-8 sentences explaining what was implemented and why.

### Post-Merge

- **Branch Cleanup**: Do **NOT** delete the feature branch after merging.
- **Environment Sync**: Check the feature worktree's `.env.local` for new variables; copy back to main if found.
