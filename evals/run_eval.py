from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp.bridge import get_bridge, reset_bridge
from uespec_mcp.llm.factory import create_llm_client
from uespec_mcp.orchestrator.core import Orchestrator


@dataclass
class EvalCase:
    category: str
    case_id: str
    title: str
    root: Path
    broken_spec_path: Path
    test_spec_path: Path
    expected_fix_path: Path
    target_file: str = "broken-spec.json"


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_temp_prefix(case_id: str) -> str:
    safe_case_id = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in case_id)
    return f"uespec-eval-{safe_case_id}-"


def load_cases(cases_root: Path, category_filter: str | None = None) -> list[EvalCase]:
    normalized_filter = category_filter.upper() if category_filter else None
    cases: list[EvalCase] = []

    for case_json_path in sorted(cases_root.rglob("case.json")):
        payload = read_json_file(case_json_path)
        category = str(payload["category"]).upper()
        if normalized_filter and category != normalized_filter:
            continue
        root = case_json_path.parent
        cases.append(
            EvalCase(
                category=category,
                case_id=str(payload["caseId"]),
                title=str(payload.get("title") or payload["caseId"]),
                root=root,
                broken_spec_path=root / "broken-spec.json",
                test_spec_path=root / "test-spec.json",
                expected_fix_path=root / "expected-fix.json",
                target_file=str(payload.get("targetFile") or "broken-spec.json"),
            )
        )

    return cases


def run_cases(
    cases: list[EvalCase],
    *,
    provider: str | None,
    max_attempts: int,
    history_root: Path,
    bridge_mode: str | None = None,
) -> dict[str, Any]:
    if bridge_mode:
        os.environ["UESPEC_BRIDGE_MODE"] = bridge_mode
        reset_bridge()
    llm_client = create_llm_client(provider)
    bridge = get_bridge()
    results: list[dict[str, Any]] = []

    for case in cases:
        case_result = run_case(
            case,
            bridge=bridge,
            llm_client=llm_client,
            max_attempts=max_attempts,
            history_root=history_root,
        )
        results.append(case_result)

    return build_eval_summary(results, provider=provider or os.getenv("UESPEC_LLM_PROVIDER", "claude"))


def run_case(
    case: EvalCase,
    *,
    bridge: Any,
    llm_client: Any,
    max_attempts: int,
    history_root: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=make_temp_prefix(case.case_id)) as temp_dir:
        workspace = Path(temp_dir)
        spec_copy = workspace / case.target_file
        shutil.copy2(case.broken_spec_path, workspace / "broken-spec.json")
        shutil.copy2(case.test_spec_path, workspace / "test-spec.json")
        shutil.copy2(case.expected_fix_path, workspace / "expected-fix.json")
        mock_test_result_path = case.root / "mock-test-result.json"
        if mock_test_result_path.is_file():
            shutil.copy2(mock_test_result_path, workspace / "mock-test-result.json")
        if case.target_file != "broken-spec.json":
            spec_copy = workspace / case.target_file

        fixture = validate_case_fixture(
            case,
            bridge=bridge,
            workspace=workspace,
        )
        if not fixture["valid"]:
            return {
                "caseId": case.case_id,
                "title": case.title,
                "category": case.category,
                "status": "invalid_fixture",
                "fixed": False,
                "attemptCount": 0,
                "historyDir": str((history_root / case.case_id).resolve()),
                "inputTokens": 0,
                "outputTokens": 0,
                "estimatedUsd": None,
                "fixtureValid": False,
                "fixtureStatus": fixture["status"],
                "fixture": fixture,
                "result": {
                    "ok": False,
                    "status": "invalid_fixture",
                    "attemptCount": 0,
                    "attempts": [],
                    "historyDir": str((history_root / case.case_id).resolve()),
                    "fixture": fixture,
                },
            }

        orchestrator = Orchestrator(
            bridge,
            llm_client,
            max_attempts=max_attempts,
            history_dir=history_root / case.case_id,
        )
        result = orchestrator.run_with_autofix(workspace / "broken-spec.json", workspace / "test-spec.json")

        expected_document = read_json_file(case.expected_fix_path)
        actual_document = read_json_file(spec_copy)
        token_totals = collect_token_totals(result)
        estimated_usd = estimate_usd_cost(token_totals)

        return {
            "caseId": case.case_id,
            "title": case.title,
            "category": case.category,
            "status": result["status"],
            "fixed": bool(result["ok"]) and expected_document == actual_document,
            "attemptCount": result["attemptCount"],
            "historyDir": result["historyDir"],
            "inputTokens": token_totals["inputTokens"],
            "outputTokens": token_totals["outputTokens"],
            "estimatedUsd": estimated_usd,
            "fixtureValid": True,
            "fixtureStatus": fixture["status"],
            "fixture": fixture,
            "result": result,
        }


def validate_case_fixture(
    case: EvalCase,
    *,
    bridge: Any,
    workspace: Path,
) -> dict[str, Any]:
    broken_check = evaluate_fixture_spec(
        bridge,
        workspace / "broken-spec.json",
        workspace / "test-spec.json",
        runtime_stem="broken-baseline",
    )
    expected_fix_check = evaluate_fixture_spec(
        bridge,
        workspace / "expected-fix.json",
        workspace / "test-spec.json",
        runtime_stem="expected-fix",
    )

    issues: list[dict[str, str]] = []
    if broken_check["ok"]:
        issues.append(
            {
                "code": "BROKEN_SPEC_ALREADY_PASSES",
                "message": "broken-spec.json already passes test-spec.json without autofix.",
            }
        )
    if not expected_fix_check["ok"]:
        issues.append(
            {
                "code": "EXPECTED_FIX_DOES_NOT_PASS",
                "message": expected_fix_check["failureMessage"] or "expected-fix.json does not pass compile + test.",
            }
        )

    status = "valid"
    if issues:
        status = issues[0]["code"].lower() if len(issues) == 1 else "multiple_fixture_issues"

    return {
        "valid": not issues,
        "status": status,
        "issues": issues,
        "brokenSpec": summarize_fixture_check(broken_check),
        "expectedFix": summarize_fixture_check(expected_fix_check),
    }


def evaluate_fixture_spec(
    bridge: Any,
    spec_path: Path,
    test_spec_path: Path,
    *,
    runtime_stem: str,
) -> dict[str, Any]:
    compile_result = bridge.compile_spec(spec_path, "/Game/UESpec/Generated")
    generated_assets = compile_result.get("generatedAssets") or []
    compile_ok = bool(compile_result.get("ok")) and bool(generated_assets)
    if not compile_ok:
        return {
            "ok": False,
            "compileOk": bool(compile_result.get("ok")),
            "testOk": False,
            "generatedAsset": generated_assets[0] if generated_assets else None,
            "failureMessage": first_compile_failure_message(compile_result),
        }

    test_document = read_json_file(test_spec_path)
    test_document.setdefault("testSpec", {})["target"] = generated_assets[0]
    runtime_test_path = spec_path.parent / f"{runtime_stem}.runtime.json"
    runtime_report_path = spec_path.parent / f"{runtime_stem}.junit.xml"
    runtime_failure_path = spec_path.parent / f"{runtime_stem}.failure-report.json"
    runtime_test_path.write_text(json.dumps(test_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_result = bridge.run_test_spec(runtime_test_path, runtime_report_path, runtime_failure_path)

    return {
        "ok": bool(test_result.get("ok")),
        "compileOk": True,
        "testOk": bool(test_result.get("ok")),
        "generatedAsset": generated_assets[0],
        "failureMessage": first_test_failure_message(test_result),
    }


def summarize_fixture_check(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(check.get("ok")),
        "compileOk": bool(check.get("compileOk")),
        "testOk": bool(check.get("testOk")),
        "generatedAsset": check.get("generatedAsset"),
        "failureMessage": str(check.get("failureMessage") or ""),
    }


def first_compile_failure_message(compile_result: dict[str, Any]) -> str:
    errors = compile_result.get("errors") or []
    if errors:
        first_error = errors[0]
        if isinstance(first_error, dict):
            return str(first_error.get("message") or first_error)
        return str(first_error)
    if not (compile_result.get("generatedAssets") or []):
        return "Compile succeeded without generated assets."
    return ""


def first_test_failure_message(test_result: dict[str, Any]) -> str:
    failure_report = test_result.get("failureReport") or {}
    failures = failure_report.get("failures") or []
    if failures:
        first_failure = failures[0]
        if isinstance(first_failure, dict):
            return str(first_failure.get("failureMessage") or first_failure)
        return str(first_failure)

    errors = test_result.get("errors") or []
    if errors:
        first_error = errors[0]
        if isinstance(first_error, dict):
            return str(first_error.get("message") or first_error)
        return str(first_error)

    return ""


def collect_token_totals(result: dict[str, Any]) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    for attempt in result.get("attempts", []):
        proposal = attempt.get("proposal") or {}
        llm_response = proposal.get("llmResponse") or {}
        input_tokens += int(llm_response.get("input_tokens") or llm_response.get("inputTokens") or 0)
        output_tokens += int(llm_response.get("output_tokens") or llm_response.get("outputTokens") or 0)
    return {"inputTokens": input_tokens, "outputTokens": output_tokens}


def estimate_usd_cost(token_totals: dict[str, int]) -> float | None:
    input_rate = os.getenv("UESPEC_LLM_INPUT_COST_PER_1K")
    output_rate = os.getenv("UESPEC_LLM_OUTPUT_COST_PER_1K")
    if not input_rate or not output_rate:
        return None
    try:
        input_cost = (token_totals["inputTokens"] / 1000.0) * float(input_rate)
        output_cost = (token_totals["outputTokens"] / 1000.0) * float(output_rate)
    except ValueError:
        return None
    return round(input_cost + output_cost, 6)


def _path_prefix_replacements() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str | None]] = [
        ("<TEMP>", os.getenv("TEMP")),
        ("<TEMP>", os.getenv("TMP")),
        ("<TEMP>", os.getenv("TMPDIR")),
        ("<TEMP>", tempfile.gettempdir()),
        ("<HOME>", str(Path.home())),
    ]

    replacements: dict[str, str] = {}
    for placeholder, raw_path in candidates:
        if not raw_path:
            continue
        variants = {raw_path, os.path.abspath(raw_path)}
        for variant in tuple(variants):
            variants.add(variant.replace("/", "\\"))
            variants.add(variant.replace("\\", "/"))
        for variant in variants:
            if variant and variant not in (".", os.curdir):
                replacements[variant] = placeholder

    return sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)


def sanitize_path_text(value: str) -> str:
    sanitized = value
    for prefix, placeholder in _path_prefix_replacements():
        sanitized = sanitized.replace(prefix, placeholder)
    return sanitized


def sanitize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str):
        return sanitize_path_text(value)
    return value


def build_eval_summary(case_results: list[dict[str, Any]], *, provider: str) -> dict[str, Any]:
    total_cases = len(case_results)
    eligible_results = [item for item in case_results if item.get("fixtureValid", True)]
    fixed_cases = sum(1 for item in eligible_results if item["fixed"])
    eligible_cases = len(eligible_results)
    invalid_fixture_cases = total_cases - eligible_cases
    total_attempts = sum(int(item["attemptCount"]) for item in eligible_results)
    total_input_tokens = sum(int(item["inputTokens"]) for item in case_results)
    total_output_tokens = sum(int(item["outputTokens"]) for item in case_results)
    estimated_usd_values = [item["estimatedUsd"] for item in case_results if item["estimatedUsd"] is not None]
    category_summary = build_category_summary(case_results)

    return {
        "provider": provider,
        "totalCases": total_cases,
        "eligibleCases": eligible_cases,
        "invalidFixtureCases": invalid_fixture_cases,
        "fixedCases": fixed_cases,
        "successRate": round((fixed_cases / eligible_cases) * 100.0, 2) if eligible_cases else 0.0,
        "avgAttempts": round(total_attempts / eligible_cases, 2) if eligible_cases else 0.0,
        "inputTokens": total_input_tokens,
        "outputTokens": total_output_tokens,
        "estimatedUsd": round(sum(estimated_usd_values), 6) if estimated_usd_values else None,
        "categorySummary": category_summary,
        "cases": case_results,
    }


def build_category_summary(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = sorted({item["category"] for item in case_results})
    summary: list[dict[str, Any]] = []
    for category in categories:
        items = [item for item in case_results if item["category"] == category]
        eligible_items = [item for item in items if item.get("fixtureValid", True)]
        fixed = sum(1 for item in eligible_items if item["fixed"])
        avg_attempts = (
            round(sum(int(item["attemptCount"]) for item in eligible_items) / len(eligible_items), 2)
            if eligible_items
            else 0.0
        )
        estimated_usd_values = [item["estimatedUsd"] for item in items if item["estimatedUsd"] is not None]
        summary.append(
            {
                "category": category,
                "total": len(items),
                "eligible": len(eligible_items),
                "invalidFixtures": len(items) - len(eligible_items),
                "fixed": fixed,
                "rate": round((fixed / len(eligible_items)) * 100.0, 2) if eligible_items else 0.0,
                "avgAttempts": avg_attempts,
                "estimatedUsd": round(sum(estimated_usd_values), 6) if estimated_usd_values else None,
            }
        )
    return summary


def summarize_report_case(case: dict[str, Any] | None) -> dict[str, Any] | str:
    if not case:
        return "No case."
    keys = ("caseId", "status", "fixed", "attemptCount", "inputTokens", "outputTokens")
    return {key: case.get(key) for key in keys}


def render_markdown_report(summary: dict[str, Any], template_path: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    category_lines = [
        "| Category | Total | Fixed | Rate | Avg Attempts | Estimated USD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["categorySummary"]:
        category_lines.append(
            f"| {entry['category']} | {entry['total']} | {entry['fixed']} | {entry['rate']}% | {entry['avgAttempts']} | {entry['estimatedUsd'] if entry['estimatedUsd'] is not None else 'n/a'} |"
        )

    best_case = next((item for item in summary["cases"] if item.get("fixtureValid", True) and item["fixed"]), None)
    worst_case = next((item for item in summary["cases"] if item.get("fixtureValid", True) and not item["fixed"]), None)
    if worst_case is None:
        worst_case = next((item for item in summary["cases"] if not item.get("fixtureValid", True)), None)

    notes = [
        f"Eligible cases: {summary.get('eligibleCases', summary['totalCases'])}/{summary['totalCases']}.",
    ]
    invalid_cases = [item for item in summary["cases"] if not item.get("fixtureValid", True)]
    if invalid_cases:
        invalid_lines = [
            f"{item['caseId']} ({item.get('fixtureStatus', 'invalid_fixture')})"
            for item in invalid_cases
        ]
        notes.append("Invalid fixtures excluded from success rate: " + ", ".join(invalid_lines) + ".")
    notes.append("Run this script manually with a real LLM provider to generate the weekly Round 5 eval snapshot.")

    replacements = {
        "{{ provider }}": str(summary["provider"]),
        "{{ total_cases }}": str(summary["totalCases"]),
        "{{ fixed_cases }}": str(summary["fixedCases"]),
        "{{ success_rate }}": f"{summary['successRate']}%",
        "{{ avg_attempts }}": str(summary["avgAttempts"]),
        "{{ input_tokens }}": str(summary["inputTokens"]),
        "{{ output_tokens }}": str(summary["outputTokens"]),
        "{{ estimated_usd }}": str(summary["estimatedUsd"] if summary["estimatedUsd"] is not None else "n/a"),
        "{{ category_table }}": "\n".join(category_lines),
        "{{ best_case }}": json.dumps(summarize_report_case(best_case), ensure_ascii=False, indent=2) if best_case else "No fixed case.",
        "{{ worst_case }}": json.dumps(summarize_report_case(worst_case), ensure_ascii=False, indent=2) if worst_case else "No failed case.",
        "{{ notes }}": "\n".join(notes),
    }

    markdown = template
    for key, value in replacements.items():
        markdown = markdown.replace(key, value)
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual Round 5 autofix evaluations.")
    parser.add_argument("--cases-root", default=str(Path(__file__).parent / "cases"), help="Root folder that contains evaluation case directories.")
    parser.add_argument("--provider", default=None, help="LLM provider override.")
    parser.add_argument("--category", default=None, help="Optional category filter.")
    parser.add_argument("--bridge-mode", default=None, choices=["commandlet", "mock", "socket"], help="Bridge mode override. Defaults to UESPEC_BRIDGE_MODE or commandlet.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum autofix attempts per case.")
    parser.add_argument("--history-dir", default="Saved/UESpec/Evals/PatchHistory", help="Patch history output directory.")
    parser.add_argument("--output-json", default=str(Path(__file__).parent / "last-results.json"), help="JSON summary output path.")
    parser.add_argument("--output-report", default=str(Path(__file__).parent / "last-report.md"), help="Markdown report output path.")
    parser.add_argument("--sanitize-paths", dest="sanitize_paths", action="store_true", default=True, help="Sanitize local home/temp paths in committed eval outputs. Enabled by default.")
    parser.add_argument("--no-sanitize-paths", dest="sanitize_paths", action="store_false", help="Write raw local paths to eval outputs.")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases_root), category_filter=args.category)
    summary = run_cases(
        cases,
        provider=args.provider,
        max_attempts=args.max_attempts,
        history_root=Path(args.history_dir),
        bridge_mode=args.bridge_mode,
    )
    if args.sanitize_paths:
        summary = sanitize_paths(summary)

    output_json_path = Path(args.output_json).expanduser().resolve()
    output_report_path = Path(args.output_report).expanduser().resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    output_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_report_path.write_text(
        render_markdown_report(summary, Path(__file__).parent / "report_template.md"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
