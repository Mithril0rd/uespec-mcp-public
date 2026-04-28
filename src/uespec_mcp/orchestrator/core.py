from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..bridge import BridgeClient, normalize_error, structured_error
from ..llm.base import LLMClient
from ..tools import surface as surface_tool
from ..tools import validate as validate_tool
from ._json_io import read_json_file, read_text_file
from .handlers import (
    propose_patch_for_assertion_failed,
    propose_patch_for_compile_error,
    propose_patch_for_slate_event_not_handled,
    propose_patch_for_timeout,
    propose_patch_for_widget_missing,
)
from .handlers.widget_missing import should_target_test_spec_for_widget_missing
from .history import PatchHistoryStore
from .patch_applier import apply_patch_to_document, apply_patch_to_spec_file, restore_spec_file
from .safety import is_regression, patch_edit_distance, patch_removes_more_than_n_lines


Handler = Callable[..., dict[str, Any]]

SUPPORTED_FAILURE_HANDLERS: dict[str, Handler] = {
    "COMPILE_ERROR": propose_patch_for_compile_error,
    "WIDGET_MISSING": propose_patch_for_widget_missing,
    "ASSERTION_FAILED": propose_patch_for_assertion_failed,
    "SLATE_EVENT_NOT_HANDLED": propose_patch_for_slate_event_not_handled,
    "TIMEOUT": propose_patch_for_timeout,
}


class Orchestrator:
    def __init__(
        self,
        bridge: BridgeClient,
        llm_client: LLMClient,
        *,
        max_attempts: int = 3,
        history_dir: Path = Path("./Saved/UESpec/PatchHistory"),
        output_dir: str = "/Game/UESpec/Generated",
        patch_removal_limit: int = 20,
        support_surface: dict[str, Any] | None = None,
    ) -> None:
        self.bridge = bridge
        self.llm_client = llm_client
        self.max_attempts = max_attempts
        self.history_dir = history_dir
        self.output_dir = output_dir
        self.patch_removal_limit = patch_removal_limit
        self.support_surface = support_surface

    def run_with_autofix(self, spec_path: str | Path, test_spec_path: str | Path) -> dict[str, Any]:
        resolved_spec_path = Path(spec_path).expanduser().resolve()
        resolved_test_spec_path = Path(test_spec_path).expanduser().resolve()
        history_store = PatchHistoryStore(self.history_dir, resolved_spec_path)

        current_evaluation = self._evaluate(resolved_spec_path, resolved_test_spec_path)
        if current_evaluation["status"] == "success":
            final_result = self._build_result(
                status="success",
                spec_path=resolved_spec_path,
                test_spec_path=resolved_test_spec_path,
                history_store=history_store,
                attempts=[],
                attempt_count=0,
                final_evaluation=current_evaluation,
            )
            history_store.write_final_result(final_result)
            return final_result

        attempts: list[dict[str, Any]] = []
        previous_patch: list[dict[str, Any]] | None = None

        for attempt_index in range(1, self.max_attempts + 1):
            failure = current_evaluation.get("failure") or {}
            failure_report = current_evaluation.get("failureReport") or _make_failure_report(
                failure,
                current_evaluation.get("failureCount", 1),
            )
            history_store.write_attempt_json(attempt_index, "failure-report.json", failure_report)

            category = str(failure.get("category") or "").upper()
            if category not in SUPPORTED_FAILURE_HANDLERS:
                attempt_record = {
                    "attempt": attempt_index,
                    "status": "unsupported_failure_type",
                    "failure": failure,
                }
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="unsupported_failure_type",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=current_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result

            target_spec_path = self._select_target_spec_path(
                failure,
                resolved_spec_path,
                resolved_test_spec_path,
            )
            before_patch_snapshots = {
                str(resolved_spec_path): read_text_file(resolved_spec_path),
            }
            if resolved_test_spec_path != resolved_spec_path:
                before_patch_snapshots[str(resolved_test_spec_path)] = read_text_file(resolved_test_spec_path)

            try:
                proposal = propose_patch_from_failure(
                    spec_path=target_spec_path,
                    failure=failure,
                    llm_client=self.llm_client,
                    support_surface=self._get_support_surface(),
                )
            except Exception as exc:
                attempt_record = {
                    "attempt": attempt_index,
                    "status": "proposal_rejected",
                    "targetSpecPath": str(target_spec_path),
                    "failureCategory": category,
                    "failure": failure,
                    "errors": [
                        structured_error(
                            "HANDLER_ERROR",
                            str(exc),
                            hint="Ensure the handler returns a valid patch for the target UISpec document.",
                        )
                    ],
                }
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="proposal_rejected",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=current_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result
            history_store.write_attempt_json(attempt_index, "llm-request.json", proposal["request"])
            history_store.write_attempt_json(attempt_index, "llm-response.json", proposal["llmResponse"])
            history_store.write_attempt_json(attempt_index, "proposed-patch.json", proposal["patch"])

            attempt_record: dict[str, Any] = {
                "attempt": attempt_index,
                "status": "proposed",
                "targetSpecPath": str(target_spec_path),
                "failureCategory": category,
                "failure": failure,
                "proposal": proposal,
            }

            patch = proposal["patch"]
            if previous_patch is not None and patch_edit_distance(previous_patch, patch) < 5:
                attempt_record["status"] = "stalled"
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="stalled",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=current_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result

            try:
                target_original_text = before_patch_snapshots[str(target_spec_path)]
                target_document = read_json_file(target_spec_path)
                patched_document = apply_patch_to_document(target_document, patch)
                patched_text = json.dumps(patched_document, ensure_ascii=False, indent=2) + "\n"
            except Exception as exc:
                attempt_record["status"] = "patch_rejected"
                attempt_record["errors"] = [
                    structured_error(
                        "INVALID_PATCH",
                        str(exc),
                        hint="Ensure the handler returned a valid RFC 6902 patch against the target document.",
                    )
                ]
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="patch_rejected",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=current_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result

            if patch_removes_more_than_n_lines(target_original_text, patched_text, self.patch_removal_limit):
                attempt_record["status"] = "patch_rejected"
                attempt_record["errors"] = [
                    structured_error(
                        "PATCH_REMOVES_TOO_MUCH",
                        f"Patch removes more than {self.patch_removal_limit} lines.",
                        hint="Try a more local patch and avoid large deletes.",
                    )
                ]
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="patch_rejected",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=current_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result

            apply_result = apply_patch_to_spec_file(target_spec_path, patch)
            history_store.write_attempt_json(
                attempt_index,
                "applied-spec.json",
                {
                    "specPath": apply_result["specPath"],
                    "document": apply_result["patchedDocument"],
                },
            )

            next_evaluation = self._evaluate(resolved_spec_path, resolved_test_spec_path)
            history_store.write_attempt_json(attempt_index, "test-result.json", next_evaluation["artifacts"])

            if next_evaluation["status"] == "success":
                attempt_record["status"] = "success"
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="success",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=next_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result

            if is_regression(current_evaluation["failureCount"], next_evaluation["failureCount"]):
                for snapshot_path, snapshot_text in before_patch_snapshots.items():
                    restore_spec_file(snapshot_path, snapshot_text)
                attempt_record["status"] = "regression_detected"
                attempt_record["nextFailureCount"] = next_evaluation["failureCount"]
                attempts.append(attempt_record)
                final_result = self._build_result(
                    status="regression_detected",
                    spec_path=resolved_spec_path,
                    test_spec_path=resolved_test_spec_path,
                    history_store=history_store,
                    attempts=attempts,
                    attempt_count=attempt_index,
                    final_evaluation=current_evaluation,
                )
                history_store.write_final_result(final_result)
                return final_result

            attempt_record["status"] = "applied"
            attempt_record["nextFailureCount"] = next_evaluation["failureCount"]
            attempts.append(attempt_record)
            previous_patch = patch
            current_evaluation = next_evaluation

        final_result = self._build_result(
            status="max_attempts_exceeded",
            spec_path=resolved_spec_path,
            test_spec_path=resolved_test_spec_path,
            history_store=history_store,
            attempts=attempts,
            attempt_count=self.max_attempts,
            final_evaluation=current_evaluation,
        )
        history_store.write_final_result(final_result)
        return final_result

    def _evaluate(self, spec_path: Path, test_spec_path: Path) -> dict[str, Any]:
        spec_text = read_text_file(spec_path)
        validate_result = validate_tool.validate_spec(spec_text)
        if not validate_result.get("ok"):
            errors = [normalize_error(error) for error in validate_result.get("errors", [])]
            failure = _make_compile_error_failure(errors[0], spec_path) if errors else _make_unknown_failure(spec_path, "Validation failed without structured errors.")
            failure_report = _make_failure_report(failure, max(len(errors), 1))
            return {
                "status": "failure",
                "phase": "validate",
                "failureCount": max(len(errors), 1),
                "failure": failure,
                "failureReport": failure_report,
                "artifacts": {"validateResult": validate_result},
            }

        compile_result = self.bridge.compile_spec(spec_path, self.output_dir)
        if not compile_result.get("ok"):
            errors = [normalize_error(error) for error in compile_result.get("errors", [])]
            failure = _make_compile_error_failure(errors[0], spec_path) if errors else _make_unknown_failure(spec_path, "Compile failed without structured errors.")
            failure_report = _make_failure_report(failure, max(len(errors), 1))
            return {
                "status": "failure",
                "phase": "compile",
                "failureCount": max(len(errors), 1),
                "failure": failure,
                "failureReport": failure_report,
                "artifacts": {
                    "validateResult": validate_result,
                    "compileResult": compile_result,
                },
            }

        report_dir = self.history_dir.expanduser().resolve() / "_runtime_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        runtime_test_spec_path = _prepare_runtime_test_spec(
            test_spec_path,
            compile_result,
            report_dir,
        )
        output_report = report_dir / f"{test_spec_path.stem}.junit.xml"
        failure_report_path = report_dir / f"{test_spec_path.stem}.failure-report.json"
        test_result = self.bridge.run_test_spec(runtime_test_spec_path, output_report, failure_report_path)
        failure_report = test_result.get("failureReport") or {}
        failures = failure_report.get("failures") or []
        summary = test_result.get("summary") or {}
        failure_count = int(summary.get("fail") or summary.get("failed") or len(failures) or len(test_result.get("errors") or []))

        if bool(test_result.get("ok")) and failure_count == 0 and not failures:
            return {
                "status": "success",
                "phase": "test",
                "failureCount": 0,
                "failure": None,
                "failureReport": failure_report,
                "artifacts": {
                    "validateResult": validate_result,
                    "compileResult": compile_result,
                    "testResult": test_result,
                    "runtimeTestSpecPath": str(runtime_test_spec_path),
                },
            }

        primary_failure = failures[0] if failures else _make_unknown_failure(spec_path, "Tests failed without a structured failure entry.")
        if not failures and test_result.get("errors"):
            primary_failure = _make_unknown_failure(spec_path, test_result["errors"][0]["message"])
        return {
            "status": "failure",
            "phase": "test",
            "failureCount": max(failure_count, 1),
            "failure": primary_failure,
            "failureReport": failure_report if failures else _make_failure_report(primary_failure, max(failure_count, 1)),
            "artifacts": {
                "validateResult": validate_result,
                "compileResult": compile_result,
                "testResult": test_result,
                "runtimeTestSpecPath": str(runtime_test_spec_path),
            },
        }

    def _select_target_spec_path(self, failure: dict[str, Any], spec_path: Path, test_spec_path: Path) -> Path:
        category = str(failure.get("category") or "").upper()
        if category == "WIDGET_MISSING" and test_spec_path != spec_path:
            test_spec_document = read_json_file(test_spec_path)
            if should_target_test_spec_for_widget_missing(failure, test_spec_document):
                return test_spec_path
        return spec_path

    def _get_support_surface(self) -> dict[str, Any]:
        if isinstance(self.support_surface, dict):
            return self.support_surface
        result = surface_tool.get_support_surface(format="json")
        content = result.get("content")
        return content if isinstance(content, dict) else {}

    def _build_result(
        self,
        *,
        status: str,
        spec_path: Path,
        test_spec_path: Path,
        history_store: PatchHistoryStore,
        attempts: list[dict[str, Any]],
        attempt_count: int,
        final_evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ok": status == "success",
            "status": status,
            "specPath": str(spec_path),
            "testSpecPath": str(test_spec_path),
            "attemptCount": attempt_count,
            "attempts": attempts,
            "historyDir": str(history_store.run_dir),
            "finalFailureCount": final_evaluation.get("failureCount", 0),
            "finalFailure": final_evaluation.get("failure"),
            "finalArtifacts": final_evaluation.get("artifacts", {}),
        }


def propose_patch_from_failure(
    *,
    spec_path: str | Path,
    failure: dict[str, Any],
    llm_client: LLMClient,
    support_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    category = str(failure.get("category") or "").upper()
    handler = SUPPORTED_FAILURE_HANDLERS.get(category)
    if handler is None:
        raise ValueError(f"Unsupported failure category '{category}'.")
    return handler(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )


def load_primary_failure(failure_report_payload: dict[str, Any]) -> dict[str, Any] | None:
    failures = failure_report_payload.get("failures")
    if isinstance(failures, list) and failures:
        first_failure = failures[0]
        return first_failure if isinstance(first_failure, dict) else None
    return None


def _make_compile_error_failure(error: dict[str, Any], spec_path: Path) -> dict[str, Any]:
    return {
        "category": "COMPILE_ERROR",
        "specPath": str(spec_path),
        "testStep": error.get("path") or "$",
        "failureMessage": error.get("message") or "Unknown compile error.",
        "hint": error.get("hint") or "",
        "nearestDeclared": error.get("nearestDeclared") or [],
        "compilerErrorCode": error.get("code") or "UNKNOWN_COMPILE_ERROR",
    }


def _make_unknown_failure(spec_path: Path, message: str) -> dict[str, Any]:
    return {
        "category": "UNKNOWN",
        "specPath": str(spec_path),
        "testStep": "$",
        "failureMessage": message,
    }


def _make_failure_report(failure: dict[str, Any], failure_count: int) -> dict[str, Any]:
    return {
        "summary": {
            "total": failure_count,
            "pass": 0,
            "fail": failure_count,
            "skip": 0,
            "duration": 0.0,
        },
        "failures": [failure],
    }


def _prepare_runtime_test_spec(test_spec_path: Path, compile_result: dict[str, Any], report_dir: Path) -> Path:
    generated_assets = compile_result.get("generatedAssets") or []
    generated_target = str(generated_assets[0]).strip() if generated_assets else ""
    if not generated_target:
        return test_spec_path

    test_document = read_json_file(test_spec_path)
    test_spec = test_document.get("testSpec")
    if not isinstance(test_spec, dict):
        return test_spec_path

    current_target = str(test_spec.get("target") or "").strip()
    if current_target == generated_target:
        return test_spec_path

    runtime_document = json.loads(json.dumps(test_document))
    runtime_document.setdefault("testSpec", {})
    runtime_document["testSpec"]["target"] = generated_target

    runtime_path = report_dir / f"{test_spec_path.stem}.runtime.json"
    runtime_path.write_text(json.dumps(runtime_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return runtime_path
