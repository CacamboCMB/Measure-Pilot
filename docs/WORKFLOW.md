# Agent execution workflow

Every delegated work order is processed serially against repository truth:

1. Bind repository, branch, local and remote HEAD, cleanliness, and workflow state.
2. Read the current requirement, scope, invariants, predecessor, successor, and target state from the canonical GitHub issue.
3. Review the real implementation and select the smallest repository-fitting change.
4. Freeze an explicit allowed-path set; unexpected expansion is an exception.
5. Continue mechanically while the reviewed meaning matches the delegated work order.
6. Implement only the current task, without speculative refactoring or future features.
7. Select targeted validation from the changed risk surface.
8. Reuse exact evidence only when code, tests, commands, and environment match.
9. On failure, reproduce, correct minimally, rerun direct tests, then one bounded adjacent check.
10. Block only for a current, reproducible, plausible, material, insufficiently mitigated defect.
11. Freeze file bytes, hashes, commands, environment, and results for the candidate.
12. Create one scoped commit with the expected parent.
13. Re-read the remote immediately before publication.
14. Publish only by strict fast-forward; never force, merge, rebase, or rewrite automatically.
15. Use remote CI only for a materially different environment.
16. Reconcile the canonical issue after publication.
17. Rediscover state from the exact published HEAD.
18. Require a clean checkpoint before another task.
19. Run batch tasks sequentially.
20. Stop only for scope/capability exceptions, publication contradictions, or realistic material defects.
