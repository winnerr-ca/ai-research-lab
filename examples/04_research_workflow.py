"""Tutorial 4: the research layer — grounded notes over real runs.

Run it:

    uv run python examples/04_research_workflow.py
    uv run python examples/04_research_workflow.py --updates 2  # quick smoke

Shows the grounding rules in action: an ungrounded claim is
unrepresentable, an LLM draft is born unapproved, and a conclusion
without human sign-off raises. Uses the deterministic MockProvider —
no network, no API key, no cost.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rlcore.agents.ppo.train import PpoConfig, train
from rlcore.research import (
    Approval,
    ApprovalRequiredError,
    MockProvider,
    ResearchStore,
    ResearchWorkflow,
    ingest_path,
    load_documents,
    retrieve,
)


def main() -> None:
    """Walk one grounded loop: source -> claim -> run -> conclusion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tutorial-04"))
    args = parser.parse_args()
    shutil.rmtree(args.out_dir, ignore_errors=True)

    store = ResearchStore(args.out_dir / "store")
    workflow = ResearchWorkflow(store, MockProvider())

    # Ground a claim in an ingested document (section-level citation).
    ingest_path(Path("docs/algorithms/ppo.md"), store)
    hits = retrieve("clipped surrogate objective", load_documents(store), top_k=1)
    claim = workflow.claim_from_sections("The PPO objective clips the importance ratio.", hits)
    print(
        f"claim {claim.entity_id} cites: "
        f"[{claim.source_refs[0].document_id} / {claim.source_refs[0].section}]"
    )

    # Ground an observation in a real run.
    run_dir = args.out_dir / "run"
    result = train(
        PpoConfig(seed=0, total_updates=args.updates, eval_every=0, stop_return=None),
        out_dir=run_dir,
    )
    reference = workflow.register_run(run_dir)
    observation = workflow.observe(
        f"PPO ran {result.total_env_steps} steps; final eval {result.final_eval.mean_return:.1f}.",
        (reference,),
    )
    analysis = workflow.analyze_runs(
        "Single run at a tiny budget: pipeline works; no performance claim.",
        "descriptive (n=1)",
        (observation,),
    )

    # The gate: without a human approval, concluding raises.
    try:
        workflow.conclude("The pipeline works end to end.", (analysis,))
    except ApprovalRequiredError:
        print("conclude() without approval: correctly refused")
    conclusion = workflow.conclude(
        "The pipeline works end to end.",
        (analysis,),
        approval=Approval(
            action="conclude", subject=analysis.entity_id, granted_by="tutorial-user"
        ),
    )
    print(f"conclusion {conclusion.entity_id} recorded (review_state={conclusion.review_state})")


if __name__ == "__main__":
    main()
