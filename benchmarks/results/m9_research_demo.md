# M9 research-layer demonstration record

End-to-end run of the grounded research pipeline on real artifacts:
the platform's own PPO doc, a real 2x2 tiny-budget ablation of
`normalize_advantages`, and the full entity chain with every gate
exercised. Provider: deterministic MockProvider (offline, free);
no paid service was called.

## What happened

- Ingested `docs/algorithms/ppo.md` -> 11 sections (sha256 `8ef7d37ad728...`).
- Retrieval for 'advantage normalization heuristic' returned: `9. Open problems this algorithm exposes` (0.2981), `5. Value loss, entropy, and the composed objective` (0.2095).
- A Claim citing those sections, then Question -> Hypothesis -> small Plan.
- 4 real PPO runs at an equal fixed 3-update budget:

| Variant | Seed | Final eval mean | Env steps |
|---|---|---|---|
| normalize=True | 0 | 466.3 | 6144 |
| normalize=True | 1 | 338.8 | 6144 |
| normalize=False | 0 | 344.1 | 6144 |
| normalize=False | 1 | 426.8 | 6144 |

No performance claim is made from n=2 at 3 updates; the recorded
conclusion is strictly about the pipeline mechanism.

## Gates exercised

- `conclude_without_approval`: raised: Action 'conclude' on 'analysis-dd46dd108121' requires an explicit human Approval(action='conclude', subject='ana
- `publish_without_approval`: raised: Action 'publish' on 'researchreport-6573fa42244c' requires an explicit human Approval(action='publish', subject=
- Approved conclusion and publication succeeded with recorded human
  approvals (granted_by, note, timestamp persisted in the store).

## Entity census

| Kind | Count |
|---|---|
| analysis | 1 |
| claim | 1 |
| conclusion | 1 |
| experimentplan | 1 |
| hypothesis | 1 |
| observation | 1 |
| researchquestion | 1 |
| researchreport | 1 |
| runreference | 4 |

Store root (regenerable): `outputs/m9-demo/store`. Raw record:
`m9_research_demo.json`.

## Reproduce

```sh
uv run python benchmarks/m9_research_demo.py
```
