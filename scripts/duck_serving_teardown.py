"""Run the pinned bundle's owned-server cleanup with bounded recovery and evidence.

Embedded in the notebook so the read-only Kaggle dataset need not be replaced.
The bundle's ownership checks and shutdown gate remain authoritative.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

PINNED_TEARDOWN_SHA256 = "c48368e330abf2574b155b42041bd5d42ea556d53340ccbbc5c8788acaa0eb19"


def configure(teardown: ModuleType) -> None:
    """Allow slow metrics/GPU release without weakening the terminal gate."""
    teardown.ENDPOINT_TIMEOUT_SECONDS = 3.0
    teardown.TERM_GRACE_SECONDS = 8.0
    teardown.KILL_GRACE_SECONDS = 4.0
    teardown.PORT_CLOSE_GRACE_SECONDS = 2.0
    teardown.GPU_QUERY_TIMEOUT_SECONDS = 5.0
    attempts = 2
    teardown.MAX_EXPLICIT_WAIT_SECONDS = (
        2 * attempts * teardown.ENDPOINT_TIMEOUT_SECONDS
        + teardown.TERM_GRACE_SECONDS + teardown.KILL_GRACE_SECONDS
        + teardown.PORT_CLOSE_GRACE_SECONDS + teardown.GPU_QUERY_TIMEOUT_SECONDS
        + 3 * teardown.PROCESS_POLL_SECONDS
    )
    original_capture = teardown.capture_endpoints

    def capture_endpoints(metrics_path, models_path, result):
        for attempt in range(1, attempts + 1):
            # The original function passes ENDPOINT_TIMEOUT_SECONDS explicitly.
            original_capture(metrics_path, models_path, result)
            result["endpoint_capture_attempts"] = attempt
            if result.get("final_metrics_sha256") and result.get("final_models_sha256"):
                result.pop("final_metrics_error", None)
                result.pop("final_models_error", None)
                return

    teardown.capture_endpoints = capture_endpoints


def failure_details(result: dict) -> dict:
    scan = result.get("process_scan_final_gate", {})
    return {
        "identity_errors": result.get("identity_errors", []),
        "root_conflict": scan.get("root_conflict"),
        "saved_conflicts": scan.get("saved_conflicts", []),
        "port_closed": result.get("port_closed"),
        "owned_survivors": scan.get("authorized_records", []),
        "suspect_survivors": scan.get("suspect_records", []),
        "marker_survivors": result.get("full_proc_marker_survivors", []),
        "gpu_query_error": result.get("gpu_query_error_after"),
        "gpu_rows": result.get("gpu_rows_after", []),
        "metrics_preserved": result.get("final_metrics_preserved"),
        "metrics_error": result.get("final_metrics_error"),
        "artifacts_preserved": result.get("required_artifacts_preserved"),
    }


def run(bundle_script: Path) -> None:
    digest = hashlib.sha256(bundle_script.read_bytes()).hexdigest()
    if digest != PINNED_TEARDOWN_SHA256:
        raise RuntimeError(f"Unreviewed serving teardown source: {digest}")
    spec = importlib.util.spec_from_file_location("duck_bundle_teardown", bundle_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving teardown: {bundle_script}")
    teardown = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teardown)
    configure(teardown)
    try:
        teardown.main()
    except Exception:
        result_path = teardown.working_paths(teardown.validate_working_dir())["result"]
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            print("VLLM_TEARDOWN_FAILURE " + json.dumps(failure_details(result), sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: duck_serving_teardown.py /bundle/serving_teardown.py")
    run(Path(sys.argv[1]))
