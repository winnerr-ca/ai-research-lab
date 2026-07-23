# Version 1.0 final demonstration record

All twenty mandated items, executed in one recorded pass. Small
environments and tiny budgets throughout — **engineering validation,
not scientific reproduction** — with every outcome stated as it
happened (a deliberate failure cell included, one item BLOCKED by
the environment and recorded as such, none fabricated).

Census: {'PASS': 19, 'BLOCKED': 1, 'FAIL': 0}  ·  wall time 196.5s
Commit: `138fd36eb927f501b8f4a0fb7163cb1cc222e797` (dirty: False)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Fresh installation (wheel -> clean venv) | PASS |  |
| 2 | Complete fast quality gate | PASS | ruff check .: ok / ruff format --check .: ok / mypy: ok / pytest -q: ok |
| 3 | PPO training | PASS | 2 updates, 512 steps, final eval 67.5 (run dir outputs/v1-demo/ppo-run) |
| 4 | Checkpoint saving | PASS | outputs/v1-demo/ppo-run/checkpoint.pt |
| 5 | Fresh-process resume | PASS | resumed-history-len 4 |
| 6 | Independent evaluation (rlcore-evaluate) | PASS | 2026-07-23 19:43:45,351 rlcore.cli outputs/v1-demo/ppo-run eval over 3 episodes (greedy): 83.3 +- 15.6 (min 63.0, max 101.0) |
| 7 | DQN validated stored results | PASS | validated stored multi-seed results at benchmarks/results/m5_dqn.json (commit 13046579b3aa4a39f52c97d7fa87164b4110f017) |
| 8 | SAC validated stored results | PASS | validated stored multi-seed results at benchmarks/results/m6_sac.json (commit 1bdd52a4e29e26025b510cb7388f598cc94566eb) |
| 9 | Multi-seed benchmark execution | PASS | 3 cells: statuses ['failed', 'ok'] — the NoSuchEnv cell is a deliberate failure, recorded as a first-class result |
| 10 | Machine-readable result generation | PASS | results.jsonl + aggregate.json + environment.json |
| 11 | Plot and report generation | PASS | report.md + 1 plot(s) |
| 12 | Paper ingestion (source preserved, SHA-256) | PASS | doc-edb4addeef30: 9 sections, sha 4fb95c8ea8d5 |
| 13 | Source-grounded explanation | PASS | cites ['7. Open problems this algorithm exposes', 'SAC (Soft Actor-Critic)']; LLM draft born review_state=draft |
| 14 | Structured experiment plan | PASS | experimentplan-208d2dbb1059 (cost_class=small, linked to hypothesis-b2d1d6e46493) |
| 15 | Plan-to-run linking | PASS | plan v2 links run 20260723-194334-ppo-cartpole-v1-s0-2f97af |
| 16 | Result analysis including failed runs | PASS | analysis-766c959e59e8 covers failed cell dqn-nosuchenv-v0-s0 |
| 17 | Evidence-grounded report (citations + run IDs) | PASS | researchreport-0e70c693cd22 v2 approved+published; claim cites '7. Open problems this algorithm exposes'; run 20260723-194334-ppo-cartpole-v1-s0-2f97af |
| 18 | Docker execution | BLOCKED | environment network policy denies all container-registry blob CDNs (Docker Hub / ECR Public / GHCR each attempted; see benchmarks/results/m8_docker.md). Daemon runs; base image unpullable. Last error tail: sHmF2-0o9kexL~i1w~t8uDS-aTDhFbJqsGZNJqln6YqpuDe4J-Udzxy0vTCkM9RBh-GWwGZkcMShIHxX9SSUgPFiRPwmQIlmX7I2u9rTWV1JUq9vaOMzj2wnBe7hZFQwCvBi3dMLIHdb9WdOT9AeuSIXsb4p8zg__&Key-Pair-Id=K2C9XPB6FLAKUF": Forbidden |
| 19 | Package build | PASS | rlcore-1.0.0-py3-none-any.whl, rlcore-1.0.0.tar.gz |
| 20 | Documentation build (mkdocs --strict) | PASS | INFO    -  Cleaning site directory INFO    -  Building documentation to directory: /home/user/ai-research-lab/site INFO    -  Documentation built in 0.34 seconds |

Raw record: `v1_demonstration.json`. Demo artifacts (regenerable):
`outputs/v1-demo/`.
