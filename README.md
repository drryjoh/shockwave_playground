# SPLAY Agent Specification Bundle

This bundle contains instructions for an offline AI coding agent to generate **SPLAY**, the Shock Playground code.

Files:

- `MASTER_PROMPT.md`: complete prompt to give the coding agent.
- `docs/requirements.md`: software and physics requirements.
- `docs/architecture.md`: recommended code architecture.
- `docs/design_history.md`: note of record summarizing the design rationale.
- `docs/tutorials.md`: target tutorial definition.
- `docs/verification_plan.md`: verification and sanity checks.

Recommended use:

1. Give `MASTER_PROMPT.md` to the coding agent first.
2. Include the `docs/` files as context or ask the agent to create equivalent files in the repository.
3. Ask the agent to implement the first working milestone: the argon target shock tutorial with five cases.

