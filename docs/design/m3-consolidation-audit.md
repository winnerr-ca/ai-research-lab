# M3 consolidation audit

**Status:** Completed — decisions implemented in this milestone
**Inputs:** the two working algorithm implementations (REINFORCE, M1; PPO, M2)
**Rule applied:** extract only where the audit demonstrates duplicated,
stable behavior; keep algorithm-specific structure local; defer anything
whose second consumer is speculative.

Verdict summary: **4 extractions, 1 partial, 4 kept local, 2 deferred.**

---

## Audit entries

### 1. MLP construction + orthogonal initialization — **EXTRACT**

| # | Finding |
|---|---|
| Duplication | REINFORCE `policy.py` inlined layer building + orthogonal init; PPO `model.py` had the same logic as `_mlp()` |
| Users | REINFORCE policy net; PPO policy and value nets |
| Variation to preserve | Output-layer gain (0.01 policy / 1.0 value); hidden widths; empty-hidden (single linear) case |
| Smallest interface | `rlcore.nets.build_mlp(in_dim, hidden_sizes, out_dim, *, final_gain)` |
| Cost of local | Three copies by M5 (DQN); init details are empirically load-bearing (Engstrom et al. 2020) and drift here is silent |
| Cost of extracting | Near zero — pure function, already identical. One real constraint: construction/init order fixes the global-RNG draw sequence, so the order is documented as part of the contract |
| Decision | **Extract.** Behavior-preserving: identical op order → bitwise-identical seeded weights (verified by validation reproduction below) |

### 2. Categorical log-prob / entropy math — **EXTRACT**

| # | Finding |
|---|---|
| Duplication | `gather(1, a.unsqueeze(1)).squeeze(1)` and `-(exp(lp)*lp).sum(-1)` appeared byte-identically in both algorithms |
| Users | REINFORCE `action_log_prob`/`entropy`; PPO `evaluate_actions`/`act` |
| Variation to preserve | None — the expressions were identical; helpers take log-probs so PPO keeps its single `log_softmax` call |
| Smallest interface | `rlcore.nets.gather_log_prob(log_probs, action)`, `rlcore.nets.entropy_from_log_probs(log_probs)` |
| Cost of local | Small but recurring; the gather/squeeze indexing is an easy off-by-one-dim site |
| Cost of extracting | Near zero; same ops on same inputs → bitwise identical |
| Decision | **Extract.** |

### 3. Action **sampling** — **DEFER** (audit's most instructive finding)

| # | Finding |
|---|---|
| Duplication | Both sample via `torch.multinomial(probs, 1, generator)` — but REINFORCE computes `probs = softmax(logits)` while PPO computes `probs = exp(log_softmax(logits))` |
| Users | Both `act()` methods |
| Variation to preserve | The exact floating-point op sequence: the two prob computations are equal in exact arithmetic but **not bitwise** in floats. Unifying them would perturb multinomial thresholds by ~1 ulp, and over ~10⁵–10⁶ draws per run the probability that at least one draw flips — cascading the entire subsequent trajectory — is not negligible |
| Smallest interface | Would be `sample_categorical(logits, generator)` |
| Cost of local | Two 3-line snippets |
| Cost of extracting | Breaks the M3 gate requirement that both algorithms reproduce their previous validation results bit-for-bit |
| Decision | **Defer** to the next milestone that already schedules a full re-validation (M4 or M5); unify to one op order then. Documented in both `act()` docstrings |

### 4. Environment creation + seeding layout + space validation — **EXTRACT**

| # | Finding |
|---|---|
| Duplication | Identical ~20-line block in both trainers: two `gym.make`, child-seed layout (0 → train env, 1 → action space, 2 → eval env), Box/Discrete validation, dim extraction |
| Users | Both trainers; every future algorithm |
| Variation to preserve | Only the error-message prefix (dropped; messages now name the env id) |
| Smallest interface | `rlcore.envs.make_discrete_env_pair(env_id, rng) -> EnvPair(train_env, eval_env, obs_dim, n_actions)` |
| Cost of local | The child-seed layout is a *cross-algorithm comparability contract* — per-algorithm copies of a protocol are how protocols silently fork |
| Cost of extracting | Near zero; identical call order preserved |
| Decision | **Extract.** The strongest candidate in the audit: the point is protocol integrity, not line count |

### 5. Evaluation loop + statistics — **EXTRACT**

| # | Finding |
|---|---|
| Duplication | Two `EvalStats` dataclasses (one with min/max, one without) and two evaluation loops (REINFORCE's via `collect_episode`, PPO's inline) |
| Users | Both trainers and both validation scripts |
| Variation to preserve | The actors' different `act` signatures (tensor vs. triple) and the greedy/stochastic + RNG-stream choice |
| Smallest interface | `rlcore.evaluation.evaluate_policy(env, select_action, *, episodes)` — a **callable**, not a policy protocol: callers close over model, determinism, and generator. Unified `EvalStats` with mean/std/min/max/lengths |
| Cost of local | Protocol drift already visible (the two `EvalStats` had diverged); a third copy coming at M5 |
| Cost of extracting | Small; greedy evaluation consumes no RNG in either implementation and the stochastic closures preserve each algorithm's exact sampling ops, so results are unchanged |
| Decision | **Extract.** A policy Protocol was considered and rejected as premature — the callable is smaller and dodges the act-signature mismatch entirely |

### 6. Run-output writing + provenance metadata — **EXTRACT (partial)**

| # | Finding |
|---|---|
| Duplication | Identical run-directory writing (config.yaml / metrics.jsonl / result.json) in both trainers; identical `git_commit()` + version-stamp dicts in both validation scripts |
| Users | Both trainers, both scripts |
| Variation to preserve | The result payload's fields (algorithm-specific by design) |
| Smallest interface | `rlcore.experiments.write_run_outputs(out_dir, config, history, result_payload)`, `git_commit()`, `run_metadata()` |
| Cost of local | Provenance stamping is exactly where copies drift into inconsistent records |
| Cost of extracting | Near zero |
| Decision | **Extract the shared mechanics.** The Markdown report *templates* stay in the scripts — they are M4 experiment-manager material, and templating them now would be speculative design |

### 7. Episodic vs. fixed-horizon collectors — **KEEP LOCAL**

| # | Finding |
|---|---|
| Duplication | Superficial only: both convert obs → tensor, act, step, record. The structure differs at every decision point |
| Users | REINFORCE (`collect_episode`: variable length, ends at boundary, no extras) vs. PPO (`RolloutCollector`: fixed horizon, crosses boundaries, stateful across calls, records logprob/value, owns the `next_values` bootstrap contract) |
| Variation a shared abstraction must preserve | Boundary semantics, statefulness, per-step extras, bootstrap computation — i.e., everything |
| Smallest interface | Would require injected callbacks for extras and a boundary-policy parameter — larger than either implementation it would replace |
| Cost of local | Two readable, separately tested collectors |
| Cost of extracting | A hook-bearing framework component — precisely the failure mode ARCHITECTURE §3 warns against |
| Decision | **Keep local.** Revisit only if M5's off-policy collection reveals a genuinely shared core |

### 8. Training loops — **KEEP LOCAL**

| # | Finding |
|---|---|
| Duplication | Shared *shape* (seed → envs → model → loop → periodic eval → early stop → final eval → write), fully shared *content* only at the edges — and the edges are what M3 extracted (envs, evaluation, run outputs) |
| Users | Both trainers |
| Variation to preserve | Collection strategy, update structure (single step vs. epochs), metric sets, stop bookkeeping |
| Smallest interface | A TrainingEngine parameterized over collect/update/metrics — the universal engine explicitly excluded from this milestone |
| Cost of local | ~60 structurally similar lines per algorithm |
| Cost of extracting | Premature generalization from n=2 with known n=4 coming (DQN, SAC have different loop shapes: replay, gradient steps per env step) |
| Decision | **Keep local.** Re-audit after M5, when four loop shapes exist |

### 9. `TrainResult` dataclasses — **KEEP LOCAL**

Near-identical fields (policy/model naming, `total_episodes` only in
REINFORCE). Unifying forces a generic artifact name and buys nothing until
checkpointing (M4) formalizes run artifacts anyway. **Keep local; fold into
the M4 checkpoint/run design.**

### 10. Hydra `main()` boilerplate — **DEFER**

~15 identical lines per trainer around `@hydra.main`. Extraction would hide
the entry point behind indirection to save two small functions; a third
algorithm (M5) makes the pattern's stability clear. **Defer.**

### 11. REINFORCE `discounted_returns` vs. PPO GAE — **KEEP LOCAL (not duplication)**

Different estimators, not copies (reward-to-go without bootstrap vs. GAE
with the `next_values` contract). The roadmap's original notion of moving
"transforms" to a shared `data` module fails the rule of two: each function
has exactly one consumer. **Keep local.**

---

## Repository structure: before → after

```
src/rlcore/                          src/rlcore/
├── types.py                         ├── types.py
├── _testing.py                      ├── _testing.py
│                                    ├── nets.py          [NEW: build_mlp, gather_log_prob,
│                                    │                          entropy_from_log_probs]
│                                    ├── envs.py          [NEW: EnvPair, make_discrete_env_pair]
│                                    ├── evaluation.py    [NEW: EvalStats, evaluate_policy]
│                                    ├── experiments.py   [NEW: write_run_outputs, git_commit,
│                                    │                          run_metadata]
├── utils/seeding.py                 ├── utils/seeding.py (unchanged)
└── agents/                          └── agents/
    ├── reinforce/                       ├── reinforce/
    │   ├── policy.py  (inline MLP)      │   ├── policy.py  (uses nets; act sampling local)
    │   ├── returns.py                   │   ├── returns.py (unchanged)
    │   ├── rollout.py (+eval,stats)     │   ├── rollout.py (collect_episode only)
    │   └── train.py   (env setup,       │   └── train.py   (uses envs/evaluation/experiments)
    │                   eval, writing)   │
    └── ppo/                             └── ppo/
        ├── model.py   (_mlp local)          ├── model.py   (uses nets; act sampling local)
        ├── gae.py                           ├── gae.py     (unchanged)
        ├── loss.py                          ├── loss.py    (+explained_variance diagnostic)
        ├── rollout.py (+eval,stats)         ├── rollout.py (collector only)
        └── train.py   (env setup,           └── train.py   (uses envs/evaluation/experiments;
                        eval, writing)                       logs explained_variance)
```

## API changes and migration notes

Breaking (pre-1.0, no deprecation shims):

| Old | New |
|---|---|
| `rlcore.agents.reinforce.rollout.evaluate(env, policy, episodes=, deterministic=, generator=)` | `rlcore.evaluation.evaluate_policy(env, select_action, episodes=)` with the model/determinism/RNG closed over in `select_action` |
| `rlcore.agents.ppo.rollout.evaluate(...)` | same as above |
| `rlcore.agents.reinforce.rollout.EvalStats` / `rlcore.agents.ppo.rollout.EvalStats` | `rlcore.evaluation.EvalStats` (superset: mean/std/min/max + lengths) |
| Env setup inlined in `train()` | `rlcore.envs.make_discrete_env_pair(env_id, rng)`; space-mismatch error messages now name the env id instead of the algorithm |
| Run-dir writing inlined in `train()` | `rlcore.experiments.write_run_outputs(...)`; file names and formats unchanged |
| `git_commit()` in each benchmark script | `rlcore.experiments.git_commit()` / `run_metadata()` |

Unchanged: every algorithm-facing entry point (`ReinforceConfig`,
`PpoConfig`, both `train()`s, `reinforce_update`, `ppo_update`,
`collect_episode`, `RolloutCollector`, `compute_gae`, all loss functions),
run-directory file formats, the seeding layout, and all hyperparameter
defaults. `explained_variance` was added to PPO's per-update metrics
(diagnostic only, observational, never part of a loss).

## Behavior-preservation verification

The gate requirement — both algorithms reproduce their previous validation
behavior — is checked by re-running both validation scripts after the
refactor and comparing the machine-readable results against the committed
M1/M2 records field by field (excluding wall-clock time and commit SHA).
Every extraction above was screened for RNG-stream neutrality first; the one
candidate that failed that screen (action sampling, entry 3) was deferred
for exactly that reason.
