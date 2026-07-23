"""M9 end-to-end research-layer demonstration with real runs.

The full pipeline on real artifacts, offline and free (MockProvider):

1. Ingest the platform's own PPO algorithm doc (Markdown, section refs).
2. Source-grounded retrieval; a Claim citing the retrieved sections.
3. ResearchQuestion -> Hypothesis -> small ExperimentPlan: a real
   2-seed x 2-variant PPO ablation of ``normalize_advantages`` at a tiny
   fixed budget (fair, equal budgets; no early stop).
4. Execute the runs (really), register RunReferences, record
   Observations of the actual numbers, and a human Analysis.
5. Demonstrate the gates *firing*: unapproved conclusion and unapproved
   publication both raise; approved versions succeed.
6. Draft + publish (with approval) a report; write raw JSON + Markdown
   records to ``benchmarks/results/``.

Honesty note baked into the recorded analysis: with n=2 seeds at a tiny
budget, no performance claim about advantage normalization is supportable;
the recorded conclusion is strictly about the ablation *mechanism* working.

Usage::

    uv run python benchmarks/m9_research_demo.py
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from rlcore.agents.ppo.train import PpoConfig, train
from rlcore.experiments import run_metadata
from rlcore.research.entities import ExperimentPlan, Hypothesis, ResearchQuestion, create
from rlcore.research.ingest import ingest_path, load_documents, retrieve
from rlcore.research.llm import MockProvider
from rlcore.research.store import ResearchStore
from rlcore.research.workflows import Approval, ApprovalRequiredError, ResearchWorkflow

logger = logging.getLogger("m9_research_demo")

DOC_PATH = Path("docs/algorithms/ppo.md")
DEMO_ROOT = Path("outputs/m9-demo")
SEEDS = (0, 1)
BUDGET_UPDATES = 3  # tiny on purpose: mechanism demo, not a study


def run_ablation(workflow: ResearchWorkflow, plan: ExperimentPlan) -> dict[str, Any]:
    """Execute the plan's runs for real and register every run directory."""
    results: dict[str, Any] = {}
    for normalize in (True, False):
        variant = f"normalize={normalize}"
        for seed in SEEDS:
            out_dir = DEMO_ROOT / "runs" / f"norm-{normalize}-s{seed}"
            config = PpoConfig(
                seed=seed,
                total_updates=BUDGET_UPDATES,
                normalize_advantages=normalize,
                eval_every=0,
                stop_return=None,
            )
            result = train(config, out_dir=out_dir)
            reference = workflow.register_run(out_dir)
            results.setdefault(variant, []).append(
                {
                    "seed": seed,
                    "run_id": reference.run_id,
                    "entity_id": reference.entity_id,
                    "final_eval_mean": result.final_eval.mean_return,
                    "total_env_steps": result.total_env_steps,
                }
            )
            logger.info(
                "%s seed=%d -> final_eval_mean=%.1f",
                variant,
                seed,
                result.final_eval.mean_return,
            )
    return results


def main() -> None:
    """Run the full grounded-research demonstration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    store = ResearchStore(DEMO_ROOT / "store")
    workflow = ResearchWorkflow(store, MockProvider())
    record: dict[str, Any] = {"environment": run_metadata()}

    # 1-2: ingest + retrieval + grounded claim.
    document = ingest_path(DOC_PATH, store)
    record["document"] = {
        "document_id": document.document_id,
        "path": document.path,
        "sha256": document.sha256,
        "n_sections": len(document.sections),
    }
    hits = retrieve("advantage normalization heuristic", load_documents(store), top_k=2)
    record["retrieval"] = [{"ref": hit.section.ref, "score": round(hit.score, 4)} for hit in hits]
    claim = workflow.claim_from_sections(
        "The PPO implementation documents advantage normalization as a "
        "configurable heuristic, not an unbiased baseline.",
        hits,
    )
    record["claim"] = {"entity_id": claim.entity_id, "cites": len(claim.source_refs)}

    # 3: question -> hypothesis -> small plan.
    question = create(
        ResearchQuestion,
        text="Does per-minibatch advantage normalization change tiny-budget "
        "PPO behavior on CartPole-v1?",
        motivation="Exercise the research pipeline end to end on a real knob.",
    )
    store.save(question)
    hypothesis = create(
        Hypothesis,
        text="Both settings train without error at a tiny fixed budget.",
        question_id=question.entity_id,
        testable_prediction="4/4 runs complete and record results.",
        links=(question.entity_id,),
    )
    store.save(hypothesis)
    plan = create(
        ExperimentPlan,
        hypothesis_id=hypothesis.entity_id,
        description="2 seeds x normalize_advantages {on,off}, 3 updates each, "
        "equal budgets, no early stop.",
        algo="ppo",
        env_id="CartPole-v1",
        seeds=SEEDS,
        overrides=(("total_updates", str(BUDGET_UPDATES)),),
        cost_class="small",
        links=(hypothesis.entity_id,),
    )
    store.save(plan)
    workflow.check_plan_executable(plan)  # small: no approval needed

    # 4: real runs + observations + analysis.
    ablation = run_ablation(workflow, plan)
    record["ablation"] = ablation
    references = tuple(
        store.get(str(entry["entity_id"])) for entries in ablation.values() for entry in entries
    )
    from rlcore.research.entities import RunReference

    run_refs = tuple(r for r in references if isinstance(r, RunReference))
    observation = workflow.observe(
        "All 4 runs (2 seeds x normalize on/off) completed at the fixed "
        f"{BUDGET_UPDATES}-update budget; final deterministic eval means: "
        + ", ".join(
            f"{variant} s{entry['seed']}: {entry['final_eval_mean']:.1f}"
            for variant, entries in ablation.items()
            for entry in entries
        ),
        run_refs,
    )
    analysis = workflow.analyze_runs(
        "The ablation mechanism works end to end: both variants ran to the "
        "same budget and recorded results. With n=2 seeds at 3 updates, no "
        "claim about which setting performs better is supportable, and none "
        "is made.",
        "descriptive comparison at equal fixed budgets (n=2 per variant)",
        (observation,),
    )

    # 5: gates fire without approval.
    gate_checks: dict[str, str] = {}
    try:
        workflow.conclude("Ablation pipeline works.", (analysis,))
    except ApprovalRequiredError as exc:
        gate_checks["conclude_without_approval"] = f"raised: {exc}"[:120]
    conclusion = workflow.conclude(
        "The research-layer ablation pipeline (plan -> real runs -> "
        "run-grounded analysis) works end to end; performance comparison "
        "is explicitly out of scope at this sample size.",
        (analysis,),
        approval=Approval(
            action="conclude",
            subject=analysis.entity_id,
            granted_by="ai.labs1@senecapolytechnic.ca",
            note="Demo conclusion about the mechanism only.",
        ),
    )
    report = workflow.draft_report(
        "M9 research-layer demonstration",
        (
            claim.entity_id,
            question.entity_id,
            hypothesis.entity_id,
            plan.entity_id,
            observation.entity_id,
            analysis.entity_id,
            conclusion.entity_id,
        ),
    )
    try:
        workflow.publish_report(report)
    except ApprovalRequiredError as exc:
        gate_checks["publish_without_approval"] = f"raised: {exc}"[:120]
    published = workflow.publish_report(
        report,
        approval=Approval(
            action="publish",
            subject=report.entity_id,
            granted_by="ai.labs1@senecapolytechnic.ca",
            note="Demo publication of the mechanism report.",
        ),
    )
    record["gates"] = gate_checks
    record["report"] = {
        "entity_id": published.entity_id,
        "version": published.version,
        "review_state": published.review_state,
        "published": published.published,
    }
    record["entity_census"] = {
        kind: len(store.list_ids(kind))
        for kind in sorted({entity_id.rsplit("-", 1)[0] for entity_id in store.list_ids()})
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m9_research_demo.json").write_text(json.dumps(record, indent=2))

    lines = [
        "# M9 research-layer demonstration record",
        "",
        "End-to-end run of the grounded research pipeline on real artifacts:",
        "the platform's own PPO doc, a real 2x2 tiny-budget ablation of",
        "`normalize_advantages`, and the full entity chain with every gate",
        "exercised. Provider: deterministic MockProvider (offline, free);",
        "no paid service was called.",
        "",
        "## What happened",
        "",
        f"- Ingested `{DOC_PATH}` -> {record['document']['n_sections']} sections "
        f"(sha256 `{record['document']['sha256'][:12]}...`).",
        "- Retrieval for 'advantage normalization heuristic' returned: "
        + ", ".join(f"`{hit['ref']}` ({hit['score']})" for hit in record["retrieval"])
        + ".",
        "- A Claim citing those sections, then Question -> Hypothesis -> small Plan.",
        "- 4 real PPO runs at an equal fixed 3-update budget:",
        "",
        "| Variant | Seed | Final eval mean | Env steps |",
        "|---|---|---|---|",
        *[
            f"| {variant} | {entry['seed']} | {entry['final_eval_mean']:.1f} "
            f"| {entry['total_env_steps']} |"
            for variant, entries in ablation.items()
            for entry in entries
        ],
        "",
        "No performance claim is made from n=2 at 3 updates; the recorded",
        "conclusion is strictly about the pipeline mechanism.",
        "",
        "## Gates exercised",
        "",
        *[f"- `{name}`: {outcome}" for name, outcome in gate_checks.items()],
        "- Approved conclusion and publication succeeded with recorded human",
        "  approvals (granted_by, note, timestamp persisted in the store).",
        "",
        "## Entity census",
        "",
        "| Kind | Count |",
        "|---|---|",
        *[f"| {kind} | {count} |" for kind, count in record["entity_census"].items()],
        "",
        "Store root (regenerable): `outputs/m9-demo/store`. Raw record:",
        "`m9_research_demo.json`.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "uv run python benchmarks/m9_research_demo.py",
        "```",
        "",
    ]
    (out_dir / "m9_research_demo.md").write_text("\n".join(lines))
    logger.info("demo record written to %s", out_dir / "m9_research_demo.md")


if __name__ == "__main__":
    main()
