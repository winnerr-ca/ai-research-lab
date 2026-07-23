# M10: Documentation, community files, release preparation (v1.0.0)

**Status:** Implemented

## Objective, and why

Turn a green codebase into a releasable project: documentation a
newcomer can actually follow (tutorials that are tested, not decorative),
the community files an open-source repository owes its users
(license/contributing/conduct/security/templates/changelog/citation),
CI that also proves the package builds and installs cleanly, and a
final report that states — with evidence — what the platform is and is
not.

## Deliverables / non-goals / acceptance

Delivered: rewritten `README.md` for v1.0; `docs/research.md`; tested
tutorials `examples/01–04` (each executed by `tests/unit/test_examples.py`
at smoke scale); `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor
Covenant 2.1), `SECURITY.md` (including the checkpoint pickle
trust-boundary note), issue/PR templates, `CHANGELOG.md`
(Keep-a-Changelog), `CITATION.cff`; `mkdocs.yml` with a strict docs
build; CI extended with build + install-smoke and docs jobs; version
1.0.0; the 20-item final demonstration and the comprehensive final
report (`docs/FINAL-REPORT.md`).

Non-goals held: **no PyPI publication and no public GitHub release**
(both require explicit user authorization — the version number and
packaging are release-*ready*, not released); no docs hosting setup; no
logo/branding work.

Acceptance: canonical gate green including the new example tests; the
wheel builds and installs into a clean venv with entry points working;
`mkdocs build --strict` green; final demonstration executed with
recorded output.

## Self-review

- Tutorials run at reduced scale in CI (minutes matter); their default
  scales are the documented behavior and were executed manually for the
  final demonstration record.
- The changelog compresses ten milestones into one 1.0.0 entry —
  appropriate for a first release; future entries go per-version.
- `CITATION.cff` names "The rlcore contributors" rather than individual
  authors; maintainers should replace this with real author metadata
  before any academic use.
