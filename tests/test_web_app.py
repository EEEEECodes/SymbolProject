"""Socket-level contract tests for the standard-library local web app."""

from __future__ import annotations

from contextlib import contextmanager
from html.parser import HTMLParser
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

import app


class _ConfigControlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.controls: set[tuple[str, str]] = set()
        self.attributes: dict[str, dict[str, str | None]] = {}
        self.picker_labels: list[str | None] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        identifier = values.get("id")
        if identifier is not None:
            self.attributes[identifier] = values
        if "data-picker" in values:
            self.picker_labels.append(values.get("aria-label"))
        group = values.get("data-config-group")
        key = values.get("data-config-key")
        if group is not None and key is not None:
            self.controls.add((group, key))


class _ArtifactBackend:
    torch = None

    @staticmethod
    def web_defaults() -> dict[str, Any]:
        return {"training": {"epochs": 3}}

    @staticmethod
    def validate_dataset(config, progress=None, cancel_event=None):
        report = Path(config["paths"]["report"])
        report.mkdir(parents=True, exist_ok=True)
        artifact = report / "summary.json"
        artifact.write_text('{"accepted": 20}', encoding="utf-8")
        if progress is not None:
            progress(0.5, "halfway", {"accepted": 10})
            progress(1.0, "done", {"accepted": 20})
        return {"status": "complete", "artifact": str(artifact)}

    train_model = validate_dataset
    generate_symbols = validate_dataset


class _BlockingBackend(_ArtifactBackend):
    started = threading.Event()

    @classmethod
    def validate_dataset(cls, config, progress=None, cancel_event=None):
        cls.started.set()
        if progress is not None:
            progress(0.1, "waiting", None)
        assert cancel_event is not None
        if not cancel_event.wait(timeout=3):
            raise TimeoutError("test job was not cancelled")
        return {"status": "cancelled"}

    train_model = validate_dataset
    generate_symbols = validate_dataset


@contextmanager
def _running_server(state: app.AppState) -> Iterator[str]:
    server = app.create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token is not None:
        headers["X-Session-Token"] = token
    if extra_headers is not None:
        headers.update(extra_headers)
    request = Request(base_url + path, data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def _json_request(*args, **kwargs) -> tuple[int, dict[str, Any]]:
    status, _headers, body = _request(*args, **kwargs)
    return status, json.loads(body.decode("utf-8"))


def _wait_for_terminal_job(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, payload = _json_request(base_url, "/api/jobs")
        assert status == 200
        if payload["job"]["status"] not in {"running", "cancelling"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("web job did not reach a terminal state")


def test_server_rejects_non_loopback_binding():
    with pytest.raises(ValueError, match="loopback"):
        app.create_server("0.0.0.0", 0, app.AppState(_ArtifactBackend))

    state = app.AppState(_ArtifactBackend, token="host-test-token")
    with _running_server(state) as base_url:
        status, payload = _json_request(
            base_url,
            "/api/bootstrap",
            extra_headers={"Host": "attacker.example"},
        )
        assert status == 421
        assert payload["ok"] is False


def test_bootstrap_token_job_and_artifact_confinement(tmp_path: Path):
    token = "fixed-test-token"
    state = app.AppState(_ArtifactBackend, token=token)
    report = tmp_path / "report"
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")

    with _running_server(state) as base_url:
        status, bootstrap = _json_request(base_url, "/api/bootstrap")
        assert status == 200
        assert bootstrap["ok"] is True
        assert bootstrap["token"] == token
        assert bootstrap["defaults"]["training"]["epochs"] == 3
        assert bootstrap["job"]["status"] == "idle"

        job_request = {
            "kind": "validate",
            "config": {"paths": {"data": str(tmp_path), "report": str(report)}},
        }
        status, rejected = _json_request(
            base_url, "/api/jobs", method="POST", payload=job_request
        )
        assert status == 403
        assert rejected["ok"] is False

        status, started = _json_request(
            base_url,
            "/api/jobs",
            method="POST",
            payload=job_request,
            token=token,
        )
        assert status == 202
        assert started["job"]["kind"] == "validate"
        terminal = _wait_for_terminal_job(base_url)
        assert terminal["job"]["status"] == "completed"
        assert any(entry["message"] == "halfway" for entry in terminal["logs"])

        status, listing = _json_request(base_url, "/api/artifacts")
        assert status == 200
        assert [item["relative"] for item in listing["files"]] == ["summary.json"]
        artifact_url = listing["files"][0]["url"]
        status, _headers, body = _request(base_url, artifact_url)
        assert status == 200
        assert json.loads(body) == {"accepted": 20}

        status, forbidden = _json_request(
            base_url, "/api/artifact?" + urlencode({"path": str(outside)})
        )
        assert status == 403
        assert forbidden["ok"] is False


def test_only_one_job_runs_and_cancel_is_cooperative(tmp_path: Path):
    _BlockingBackend.started = threading.Event()
    token = "cancel-test-token"
    state = app.AppState(_BlockingBackend, token=token)
    request_body = {
        "kind": "validate",
        "config": {"paths": {"data": str(tmp_path), "report": str(tmp_path / "report")}},
    }

    with _running_server(state) as base_url:
        status, _payload = _json_request(
            base_url,
            "/api/jobs",
            method="POST",
            payload=request_body,
            token=token,
        )
        assert status == 202
        assert _BlockingBackend.started.wait(timeout=1)

        status, busy = _json_request(
            base_url,
            "/api/jobs",
            method="POST",
            payload=request_body,
            token=token,
        )
        assert status == 409
        assert "already running" in busy["error"]

        status, cancelling = _json_request(
            base_url,
            "/api/jobs/cancel",
            method="POST",
            payload={},
            token=token,
        )
        assert status == 200
        assert cancelling["job"]["cancel_requested"] is True
        terminal = _wait_for_terminal_job(base_url)
        assert terminal["job"]["status"] == "cancelled"


def test_static_interface_has_all_primary_sections_and_security_headers():
    state = app.AppState(app.train, token="static-test-token")
    with _running_server(state) as base_url:
        status, headers, html = _request(base_url, "/")
        assert status == 200
        assert headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        page = html.decode("utf-8")
        for label in ("Dataset", "Training", "Generation", "Results", "Environment"):
            assert label in page
        for paired_copy in (
            "One base and a deviations folder per family",
            "Initialize from checkpoint",
            "Unseen-base audit",
            "Base image",
            "Family summary",
        ):
            assert paired_copy in page

        parser = _ConfigControlParser()
        parser.feed(page)
        defaults = app.train.web_defaults()
        expected_controls = {
            (group, key)
            for group, values in defaults.items()
            if group != "safety"
            for key in values
        }
        # Canvas size has one visible control.  JavaScript mirrors it into the
        # model group so the backend's cross-field invariant cannot drift.
        expected_controls.remove(("model", "image_size"))
        assert parser.controls == expected_controls
        assert ("paths", "resume") in parser.controls
        assert ("paths", "init_checkpoint") in parser.controls
        assert ("paths", "base") in parser.controls
        assert ("registration", "minimum_overlap") in parser.controls
        assert ("training", "real_pair_probability") in parser.controls
        assert ("training", "audit_sample_count") in parser.controls
        assert ("training", "empty_condition_probability") not in parser.controls
        expected_bounds = {
            "pre-validation": {"min": "0.01", "max": "0.49"},
            "reg-translation": {"min": "0", "max": "32"},
            "reg-scale": {"min": "0", "max": "0.49"},
            "reg-tolerance": {"min": "0", "max": "12"},
            "gen-threshold": {"min": "0.05", "max": "0.95"},
            "nov-scale": {"min": "0", "max": "0.49"},
            "quality-ink": {"min": "0.01", "max": "0.99"},
            "quality-distance": {"min": "0.1"},
            "quality-parallel": {"min": "2"},
        }
        for identifier, bounds in expected_bounds.items():
            for name, value in bounds.items():
                assert parser.attributes[identifier][name] == value
        assert len(parser.picker_labels) == 8
        assert None not in parser.picker_labels
        assert len(set(parser.picker_labels)) == len(parser.picker_labels)

        for path, content_type in (
            ("/app.js", "javascript"),
            ("/styles.css", "text/css"),
        ):
            status, headers, body = _request(base_url, path)
            assert status == 200
            assert content_type in headers["Content-Type"]
            assert body
            if path == "/app.js":
                script = body.decode("utf-8")
                assert "config.model.image_size = config.preprocessing.image_size" in script
                assert "paths.resume && paths.init_checkpoint" in script
                assert "training.real_pair_probability" in script
                assert "reportRelevantValidity(kind)" in script
                assert "ui.form.reportValidity()" not in script
                assert 'payload?.stage === "audit"' in script
                assert '"scores_reconstruction_dice"' in script
                assert 'findNamedData(sources, ["family_summaries"' in script
                assert 'findNamedData(sources, ["audit"' in script
