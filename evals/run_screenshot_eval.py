from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp.tools.generate import generate_spec_from_image_with_autofix


@dataclass(frozen=True)
class ScreenshotEvalCase:
    case_id: str
    title: str
    image_path: Path
    test_spec_path: Path
    target_kind: str
    visual_prompt: str
    expected_components: list[str]
    scoring_focus: list[str]


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_case_slug(case_id: str) -> str:
    slug = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in case_id)
    return slug.strip("-_") or "screenshot-case"


def load_cases(
    manifest_path: Path,
    *,
    root_dir: Path = REPO_ROOT,
    case_id_filter: str | None = None,
) -> list[ScreenshotEvalCase]:
    manifest = read_json_file(manifest_path)
    default_test_spec = manifest.get("defaultTestSpecPath")
    cases: list[ScreenshotEvalCase] = []

    for item in manifest.get("cases", []):
        case_id = str(item["caseId"])
        if case_id_filter and case_id != case_id_filter:
            continue
        test_spec_value = item.get("testSpecPath") or default_test_spec
        if not test_spec_value:
            raise ValueError(f"Screenshot eval case '{case_id}' does not define a test spec path.")

        cases.append(
            ScreenshotEvalCase(
                case_id=case_id,
                title=str(item.get("title") or case_id),
                image_path=resolve_repo_path(str(item["imagePath"]), root_dir=root_dir),
                test_spec_path=resolve_repo_path(str(test_spec_value), root_dir=root_dir),
                target_kind=str(item.get("targetKind") or "HUD"),
                visual_prompt=str(item.get("visualPrompt") or ""),
                expected_components=[str(value) for value in item.get("expectedComponents", [])],
                scoring_focus=[str(value) for value in item.get("scoringFocus", [])],
            )
        )

    return cases


def resolve_repo_path(value: str, *, root_dir: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()


def run_cases(
    cases: list[ScreenshotEvalCase],
    *,
    provider: str | None,
    max_attempts: int,
    max_examples: int,
    output_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        run_case(
            case,
            provider=provider,
            max_attempts=max_attempts,
            max_examples=max_examples,
            output_root=output_root,
            dry_run=dry_run,
        )
        for case in cases
    ]

    return build_summary(
        results,
        provider=provider or os.getenv("UESPEC_LLM_PROVIDER", "claude"),
        dry_run=dry_run,
    )


def run_case(
    case: ScreenshotEvalCase,
    *,
    provider: str | None,
    max_attempts: int,
    max_examples: int,
    output_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    slug = make_case_slug(case.case_id)
    output_root = output_root.expanduser().resolve()
    generated_spec_path = output_root / "generated_specs" / f"{slug}.generated.json"
    history_dir = output_root / "patch_history" / slug

    base_result = {
        "caseId": case.case_id,
        "title": case.title,
        "targetKind": case.target_kind,
        "imagePath": str(case.image_path),
        "testSpecPath": str(case.test_spec_path),
        "generatedSpecPath": str(generated_spec_path),
        "expectedComponents": case.expected_components,
        "scoringFocus": case.scoring_focus,
        "humanScore": None,
        "manualEditMinutesEstimate": None,
    }

    if not case.image_path.is_file():
        return {
            **base_result,
            "ok": False,
            "status": "image_missing",
            "machineUsable": False,
            "inputTokens": 0,
            "outputTokens": 0,
            "result": None,
            "errors": [
                {
                    "code": "IMAGE_NOT_FOUND",
                    "message": f"Image file '{case.image_path}' does not exist.",
                    "path": "$.imagePath",
                    "hint": "Place the screenshot at the manifest imagePath before running the real eval.",
                }
            ],
        }

    if not case.test_spec_path.is_file():
        return {
            **base_result,
            "ok": False,
            "status": "test_spec_missing",
            "machineUsable": False,
            "inputTokens": 0,
            "outputTokens": 0,
            "result": None,
            "errors": [
                {
                    "code": "TEST_SPEC_NOT_FOUND",
                    "message": f"Test spec file '{case.test_spec_path}' does not exist.",
                    "path": "$.testSpecPath",
                    "hint": "Fix the manifest testSpecPath or create the referenced test spec.",
                }
            ],
        }

    if dry_run:
        return {
            **base_result,
            "ok": False,
            "status": "dry_run",
            "machineUsable": False,
            "inputTokens": 0,
            "outputTokens": 0,
            "result": None,
            "errors": [],
        }

    generated_spec_path.parent.mkdir(parents=True, exist_ok=True)
    result = generate_spec_from_image_with_autofix(
        image_path=str(case.image_path),
        test_spec_path=str(case.test_spec_path),
        target_kind=case.target_kind,
        output_path=str(generated_spec_path),
        visual_prompt=case.visual_prompt,
        provider=provider,
        max_examples=max_examples,
        max_attempts=max_attempts,
        history_dir=str(history_dir),
    )
    token_totals = collect_token_totals(result)

    return {
        **base_result,
        "ok": bool(result.get("ok")),
        "status": str(result.get("status") or "unknown"),
        "machineUsable": bool(result.get("ok")),
        "inputTokens": token_totals["inputTokens"],
        "outputTokens": token_totals["outputTokens"],
        "result": result,
        "errors": result.get("errors", []),
    }


def collect_token_totals(result: dict[str, Any]) -> dict[str, int]:
    generation = result.get("generation") or {}
    generation_llm = generation.get("llm") or {}
    input_tokens = int(generation_llm.get("inputTokens") or generation_llm.get("input_tokens") or 0)
    output_tokens = int(generation_llm.get("outputTokens") or generation_llm.get("output_tokens") or 0)

    autofix = result.get("autofix") or {}
    for attempt in autofix.get("attempts", []):
        proposal = attempt.get("proposal") or {}
        llm_response = proposal.get("llmResponse") or {}
        input_tokens += int(llm_response.get("inputTokens") or llm_response.get("input_tokens") or 0)
        output_tokens += int(llm_response.get("outputTokens") or llm_response.get("output_tokens") or 0)

    return {"inputTokens": input_tokens, "outputTokens": output_tokens}


def build_summary(results: list[dict[str, Any]], *, provider: str, dry_run: bool) -> dict[str, Any]:
    total_cases = len(results)
    missing_inputs = sum(1 for item in results if item["status"] in {"image_missing", "test_spec_missing"})
    completed_results = [item for item in results if item["status"] not in {"image_missing", "test_spec_missing", "dry_run"}]
    usable_cases = sum(1 for item in completed_results if item["machineUsable"])
    score_counts = build_human_score_counts(results)

    return {
        "suite": "round7-screenshot-to-uispec",
        "provider": provider,
        "dryRun": dry_run,
        "totalCases": total_cases,
        "completedCases": len(completed_results),
        "missingInputCases": missing_inputs,
        "machineUsableCases": usable_cases,
        "machineUsableRate": round((usable_cases / len(completed_results)) * 100.0, 2) if completed_results else 0.0,
        "humanScoreCounts": score_counts,
        "inputTokens": sum(int(item["inputTokens"]) for item in results),
        "outputTokens": sum(int(item["outputTokens"]) for item in results),
        "cases": results,
    }


def build_human_score_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"usable": 0, "needs_small_edit": 0, "not_usable": 0, "unscored": 0}
    for item in results:
        score = item.get("humanScore")
        if score in counts:
            counts[str(score)] += 1
        else:
            counts["unscored"] += 1
    return counts


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 7 Screenshot Eval Report",
        "",
        f"Provider: `{summary['provider']}`",
        f"Dry run: `{summary['dryRun']}`",
        f"Total cases: {summary['totalCases']}",
        f"Completed cases: {summary['completedCases']}",
        f"Missing input cases: {summary['missingInputCases']}",
        f"Machine usable: {summary['machineUsableCases']} ({summary['machineUsableRate']}%)",
        f"Tokens: input={summary['inputTokens']}, output={summary['outputTokens']}",
        "",
        "## Case Table",
        "",
        "| Case | Target | Status | Machine usable | Human score | Generated spec |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in summary["cases"]:
        human_score = item.get("humanScore") or "unscored"
        generated_spec = item.get("generatedSpecPath") or ""
        lines.append(
            f"| {item['caseId']} | {item['targetKind']} | {item['status']} | {str(item['machineUsable']).lower()} | {human_score} | `{generated_spec}` |"
        )

    lines.extend(
        [
            "",
            "## Human Review Rubric",
            "",
            "- `usable`: compiles, smoke test passes, and needs only minor tuning.",
            "- `needs_small_edit`: compiles or is close, but needs manual fixes within 10 minutes.",
            "- `not_usable`: fails to produce a workable UISpec or requires major rebuild.",
            "",
            "## Per-Case Review Notes",
            "",
        ]
    )
    for item in summary["cases"]:
        lines.extend(
            [
                f"### {item['caseId']}",
                "",
                f"- Title: {item['title']}",
                f"- Expected components: {', '.join(item['expectedComponents']) or 'n/a'}",
                f"- Scoring focus: {', '.join(item['scoringFocus']) or 'n/a'}",
                "- Human score: unscored",
                "- Manual edit minutes estimate: n/a",
                "- Notes: ",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round 7 screenshot-to-UISpec evaluations.")
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "screenshot_cases" / "manifest.json"), help="Screenshot eval manifest path.")
    parser.add_argument("--provider", default=None, help="LLM provider override.")
    parser.add_argument("--case-id", default=None, help="Optional exact caseId filter.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum autofix attempts per case.")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum retrieved examples for screenshot generation.")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "artifacts" / "screenshot_eval"), help="Output directory for generated specs and reports.")
    parser.add_argument("--output-json", default=None, help="JSON summary output path. Defaults to output-dir/last-results.json.")
    parser.add_argument("--output-report", default=None, help="Markdown report output path. Defaults to output-dir/last-report.md.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest paths and produce a report without calling the LLM or bridge.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cases = load_cases(manifest_path, case_id_filter=args.case_id)
    summary = run_cases(
        cases,
        provider=args.provider,
        max_attempts=args.max_attempts,
        max_examples=args.max_examples,
        output_root=output_dir,
        dry_run=args.dry_run,
    )

    output_json_path = Path(args.output_json).expanduser().resolve() if args.output_json else output_dir / "last-results.json"
    output_report_path = Path(args.output_report).expanduser().resolve() if args.output_report else output_dir / "last-report.md"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_report_path.write_text(render_markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()

