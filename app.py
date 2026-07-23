"""Loopback-only web interface for the line-only symbol generator.

The server intentionally uses only Python's standard library.  It serves the
files in ``web/`` and adapts the public functions in :mod:`train` to a small,
polling JSON API.  It is local software, not a remotely deployable web app:
``create_server`` rejects non-loopback hosts and all state-changing requests
require a per-process session token.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import ipaddress
import json
import mimetypes
import os
import platform
import secrets
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import train
except Exception as exc:  # Keep the UI available so it can explain the failure.
    train = None  # type: ignore[assignment]
    TRAIN_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    TRAIN_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_LOG_ENTRIES = 2_000
MAX_ARTIFACTS = 1_000


FALLBACK_DEFAULTS: dict[str, Any] = {
    "paths": {
        "data": "",
        "report": str(ROOT_DIR / "validation"),
        "run": str(ROOT_DIR / "runs" / "symbols"),
        "resume": "",
        "init_checkpoint": "",
        "checkpoint": "",
        "base": "",
        "out": str(ROOT_DIR / "generated"),
    },
    "preprocessing": {
        "image_size": 128,
        "margin": 12,
        "max_source_stroke_width": 12.0,
        "min_component_pixels": 3,
        "max_input_pixels": 40_000_000,
        "filled_policy": "outline",
        "validation_fraction": 0.10,
    },
    "model": {
        "latent_dim": 32,
        "base_channels": 32,
        "min_stroke_width": 1.0,
        "max_stroke_width": 6.0,
    },
    "registration": {
        "angle_range": 12.0,
        "translation_range": 8,
        "scale_range": 0.12,
        "match_tolerance": 3.0,
        "minimum_overlap": 0.25,
    },
    "training": {
        "device": "auto",
        "epochs": 250,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "patience": 30,
        "seed": 1337,
        "beta_max": 0.001,
        "beta_warmup_fraction": 0.25,
        "real_pair_probability": 0.60,
        "synthetic_pair_probability": 0.30,
        "identity_pair_probability": 0.10,
        "delta_loss_weight": 0.50,
        "retention_loss_weight": 0.25,
        "audit_sample_count": 32,
        "gradient_clip": 1.0,
        "workers": 0,
        "deterministic": True,
        "mixed_precision": True,
        "preview_count": 8,
        "preview_frequency": 10,
    },
    "generation": {
        "count": 50,
        "edit_strength": 0.35,
        "temperature": 0.9,
        "sampling_batch": 8,
        "threshold_override": None,
        "review_cap": None,
        "attempt_multiplier": 100,
        "seed": 1337,
        "device": "auto",
    },
    "novelty": {
        "duplicate_threshold": 0.94,
        "review_threshold": 0.82,
        "transformed_review_threshold": 0.90,
        "skeleton_weight": 0.60,
        "rendered_weight": 0.30,
        "topology_weight": 0.10,
        "skeleton_tolerance": 2.0,
        "alignment_angle": 6.0,
        "alignment_translation": 3,
        "alignment_scale": 0.04,
        "shortlist_maximum": 64,
        "precise_finalists": 8,
    },
    "quality": {
        "curve_error": 0.75,
        "maximum_ink": 0.35,
        "maximum_components": 24,
        "crowded_line_limit": 0.10,
        "crowd_distance_factor": 1.5,
        "parallel_bundle_threshold": 3,
        "solid_diameter_factor": 2.2,
        "guided_noop_pixels": 8.0,
        "guided_noop_fraction": 0.08,
    },
    "safety": {
        "minimum_families": 4,
        "minimum_deviations_per_family": 20,
        "recommended_deviations_per_family": 30,
        "allowed_svg_elements": ["svg", "g", "path"],
        "stroke": "black",
        "fill": "none",
        "editable": False,
    },
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of *base* with values from *override*."""

    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = _deep_merge(value, {}) if isinstance(value, Mapping) else value
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert backend return values to a bounded JSON-compatible shape."""

    if depth > 12:
        return repr(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:2_000]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:2_000]]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item(), depth=depth + 1)
        except Exception:
            pass
    return repr(value)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _nested_value(config: Mapping[str, Any], group: str, key: str) -> Any:
    section = config.get(group)
    if isinstance(section, Mapping) and key in section:
        return section[key]
    return config.get(key)


def _resolved_path(value: Any) -> Path | None:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(candidate), str(root))) == str(root)
    except (OSError, ValueError):
        return False


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class JobManager:
    """Own one backend job and the artifact roots selected for this session."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.allowed_roots: set[Path] = set()
        self.job: dict[str, Any] = self._idle_job()
        self.logs: list[dict[str, Any]] = []
        self._next_log_index = 1

    @staticmethod
    def _idle_job() -> dict[str, Any]:
        return {
            "id": None,
            "kind": None,
            "status": "idle",
            "progress": 0.0,
            "message": "Ready",
            "payload": {},
            "result": None,
            "started_at": None,
            "ended_at": None,
            "cancel_requested": False,
        }

    def _log(self, message: Any, level: str = "info") -> None:
        text = str(message).strip()
        if not text:
            return
        with self.lock:
            self.logs.append(
                {
                    "index": self._next_log_index,
                    "time": _utc_now(),
                    "level": level,
                    "message": text,
                }
            )
            self._next_log_index += 1
            if len(self.logs) > MAX_LOG_ENTRIES:
                del self.logs[: len(self.logs) - MAX_LOG_ENTRIES]

    def _progress(self, fraction: Any = None, message: Any = None, payload: Any = None) -> None:
        if isinstance(fraction, Mapping):
            event = fraction
            fraction = event.get("fraction", event.get("progress"))
            message = event.get("message", message)
            payload = event.get("payload", payload)
        try:
            numeric = float(fraction)
            if numeric > 1.0 and numeric <= 100.0:
                numeric /= 100.0
            numeric = max(0.0, min(1.0, numeric))
        except (TypeError, ValueError):
            numeric = None

        with self.lock:
            if numeric is not None:
                self.job["progress"] = numeric
            if message is not None and str(message).strip():
                self.job["message"] = str(message).strip()
            if payload is not None:
                safe_payload = _json_safe(payload)
                self.job["payload"] = safe_payload if isinstance(safe_payload, dict) else {"value": safe_payload}
        if message is not None:
            self._log(message)

    def _register_artifact_root(self, kind: str, config: Mapping[str, Any]) -> None:
        key = {"validate": "report", "train": "run", "generate": "out"}[kind]
        root = _resolved_path(_nested_value(config, "paths", key))
        if root is not None:
            with self.lock:
                self.allowed_roots.add(root)

    def start(self, kind: str, config: Mapping[str, Any]) -> dict[str, Any]:
        if kind not in {"validate", "train", "generate"}:
            raise ValueError("kind must be validate, train, or generate")
        if not isinstance(config, Mapping):
            raise ValueError("config must be a JSON object")

        function_name = {
            "validate": "validate_dataset",
            "train": "train_model",
            "generate": "generate_symbols",
        }[kind]
        function = getattr(self.backend, function_name, None) if self.backend is not None else None
        if not callable(function):
            detail = f" ({TRAIN_IMPORT_ERROR})" if TRAIN_IMPORT_ERROR else ""
            raise RuntimeError(f"Backend function train.{function_name} is unavailable{detail}")

        with self.lock:
            if self.job["status"] in {"running", "cancelling"}:
                raise RuntimeError("Another job is already running")
            self.cancel_event = threading.Event()
            self.logs = []
            self.job = {
                "id": secrets.token_hex(6),
                "kind": kind,
                "status": "running",
                "progress": 0.0,
                "message": f"Starting {kind}",
                "payload": {},
                "result": None,
                "started_at": _utc_now(),
                "ended_at": None,
                "cancel_requested": False,
            }
            safe_config = dict(config)
            self._register_artifact_root(kind, safe_config)
            self.thread = threading.Thread(
                target=self._run,
                args=(function, safe_config),
                name=f"symbol-{kind}-{self.job['id']}",
                daemon=True,
            )
            self._log(f"{kind.capitalize()} job started")
            self.thread.start()
            snapshot = dict(self.job)
        return snapshot

    def _run(self, function: Callable[..., Any], config: dict[str, Any]) -> None:
        try:
            result = function(config, progress=self._progress, cancel_event=self.cancel_event)
            safe_result = _json_safe(result)
            with self.lock:
                cancelled = self.cancel_event.is_set()
                self.job["result"] = safe_result
                self.job["status"] = "cancelled" if cancelled else "completed"
                self.job["progress"] = self.job["progress"] if cancelled else 1.0
                self.job["message"] = "Cancelled" if cancelled else "Completed"
                self.job["ended_at"] = _utc_now()
            self._log("Job cancelled" if cancelled else "Job completed", "warning" if cancelled else "success")
        except Exception as exc:
            with self.lock:
                cancelled = self.cancel_event.is_set()
                self.job["status"] = "cancelled" if cancelled else "failed"
                self.job["message"] = "Cancelled" if cancelled else str(exc)
                self.job["result"] = {"error": f"{type(exc).__name__}: {exc}"}
                self.job["ended_at"] = _utc_now()
            if cancelled:
                self._log("Job cancelled", "warning")
            else:
                self._log(f"{type(exc).__name__}: {exc}", "error")
                for line in traceback.format_exc().rstrip().splitlines():
                    self._log(line, "debug")

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            if self.job["status"] not in {"running", "cancelling"}:
                raise RuntimeError("There is no running job to cancel")
            self.cancel_event.set()
            self.job["status"] = "cancelling"
            self.job["cancel_requested"] = True
            self.job["message"] = "Cancellation requested"
            snapshot = dict(self.job)
        self._log("Cancellation requested", "warning")
        return snapshot

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        with self.lock:
            job = _json_safe(dict(self.job))
            logs = [dict(item) for item in self.logs if item["index"] > after]
            cursor = self.logs[-1]["index"] if self.logs else after
        return {"job": job, "logs": logs, "log_cursor": cursor}

    def is_allowed_artifact(self, path: Path) -> bool:
        try:
            candidate = path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return False
        with self.lock:
            roots = tuple(self.allowed_roots)
        return candidate.is_file() and any(_is_within(candidate, root) for root in roots)

    def artifacts(self) -> list[dict[str, Any]]:
        with self.lock:
            roots = sorted(self.allowed_roots, key=lambda item: str(item).casefold())
        output: list[dict[str, Any]] = []
        for root in roots:
            if len(output) >= MAX_ARTIFACTS:
                break
            if root.is_file():
                candidates = [root]
            elif root.is_dir():
                try:
                    candidates = root.rglob("*")
                except OSError:
                    continue
            else:
                continue
            try:
                for candidate in candidates:
                    if len(output) >= MAX_ARTIFACTS:
                        break
                    try:
                        resolved = candidate.resolve(strict=True)
                        if not resolved.is_file() or not _is_within(resolved, root):
                            continue
                        relative = resolved.relative_to(root).as_posix() if root.is_dir() else resolved.name
                        stat = resolved.stat()
                    except (OSError, RuntimeError, ValueError):
                        continue
                    suffix = resolved.suffix.casefold()
                    output.append(
                        {
                            "name": resolved.name,
                            "relative": relative,
                            "root": str(root),
                            "path": str(resolved),
                            "size": stat.st_size,
                            "modified": dt.datetime.fromtimestamp(
                                stat.st_mtime, tz=dt.timezone.utc
                            ).isoformat(timespec="seconds"),
                            "preview": suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"},
                            "url": "/api/artifact?" + urllib.parse.urlencode({"path": str(resolved)}),
                        }
                    )
            except OSError:
                continue
        output.sort(key=lambda item: (item["root"].casefold(), item["relative"].casefold()))
        return output


class AppState:
    """Socket-independent application state used by the request handler."""

    def __init__(self, backend: Any = train, token: str | None = None) -> None:
        self.backend = backend
        self.token = token or secrets.token_urlsafe(32)
        self.jobs = JobManager(backend)
        self.picker_lock = threading.Lock()

    def defaults(self) -> dict[str, Any]:
        function = getattr(self.backend, "web_defaults", None) if self.backend is not None else None
        if not callable(function):
            return _deep_merge(FALLBACK_DEFAULTS, {})
        try:
            backend_defaults = function()
            if not isinstance(backend_defaults, Mapping):
                raise TypeError("web_defaults() did not return a mapping")
            return _deep_merge(FALLBACK_DEFAULTS, backend_defaults)
        except Exception:
            return _deep_merge(FALLBACK_DEFAULTS, {})

    def environment(self) -> dict[str, Any]:
        torch_module = getattr(self.backend, "torch", None) if self.backend is not None else None
        cuda_available = False
        cuda_name: str | None = None
        torch_version: str | None = None
        if torch_module is not None:
            torch_version = getattr(torch_module, "__version__", None)
            try:
                cuda_available = bool(torch_module.cuda.is_available())
                if cuda_available:
                    cuda_name = str(torch_module.cuda.get_device_name(0))
            except Exception:
                cuda_available = False
        return {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "working_directory": str(ROOT_DIR),
            "torch": torch_version or "not installed",
            "cuda_available": cuda_available,
            "cuda_device": cuda_name,
            "backend_import_error": TRAIN_IMPORT_ERROR,
            "backend_functions": {
                name: callable(getattr(self.backend, name, None)) if self.backend is not None else False
                for name in ("web_defaults", "validate_dataset", "train_model", "generate_symbols")
            },
            "server": "Python standard library, loopback only",
        }

    def bootstrap(self) -> dict[str, Any]:
        return {
            "ok": True,
            "token": self.token,
            "defaults": self.defaults(),
            "environment": self.environment(),
            **self.jobs.snapshot(),
        }

    def pick_path(self, kind: str, title: str, initial: str = "") -> str:
        if kind not in {"directory", "file"}:
            raise ValueError("kind must be directory or file")
        with self.picker_lock:
            try:
                import tkinter as tk
                from tkinter import filedialog
            except Exception as exc:
                raise RuntimeError(f"Native picker is unavailable; enter the path manually ({exc})") from exc

            root = None
            try:
                root = tk.Tk()
                root.withdraw()
                try:
                    root.attributes("-topmost", True)
                except tk.TclError:
                    pass
                initial_path = Path(initial).expanduser() if initial else ROOT_DIR
                if kind == "directory":
                    initial_dir = initial_path if initial_path.is_dir() else initial_path.parent
                    selected = filedialog.askdirectory(
                        parent=root,
                        title=title or "Choose a folder",
                        initialdir=str(initial_dir),
                        mustexist=False,
                    )
                else:
                    initial_dir = initial_path if initial_path.is_dir() else initial_path.parent
                    initial_file = "" if initial_path.is_dir() else initial_path.name
                    selected = filedialog.askopenfilename(
                        parent=root,
                        title=title or "Choose a file",
                        initialdir=str(initial_dir),
                        initialfile=initial_file,
                        filetypes=(
                            (
                                "Supported images and checkpoints",
                                "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.svg *.pt *.pth",
                            ),
                            ("All files", "*.*"),
                        ),
                    )
                return str(Path(selected).resolve(strict=False)) if selected else ""
            except Exception as exc:
                raise RuntimeError(f"Native picker failed; enter the path manually ({exc})") from exc
            finally:
                if root is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass


def create_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to *state* without opening a socket."""

    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "SymbolTrainer/1"

        def log_message(self, format_string: str, *args: Any) -> None:
            # The token is only ever in a response body, never a URL or log line.
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format_string % args))

        def _security_headers(self, *, api: bool = False, artifact: bool = False) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("X-Frame-Options", "DENY")
            if artifact:
                # Output folders can predate this run.  Treat their contents as
                # untrusted even though generated SVGs are structurally strict.
                self.send_header(
                    "Content-Security-Policy",
                    "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
                    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                )
            else:
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data: blob:; "
                    "script-src 'self'; style-src 'self'; connect-src 'self'; "
                    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
                )
            if api:
                self.send_header("Cache-Control", "no-store")

        def _send_json(self, payload: Mapping[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._security_headers(api=True)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status)

        def _parse_url(self) -> tuple[str, dict[str, list[str]]]:
            parsed = urllib.parse.urlsplit(self.path)
            return parsed.path, urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

        def _read_json(self) -> dict[str, Any]:
            length_text = self.headers.get("Content-Length", "0")
            try:
                length = int(length_text)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is too large")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request body must be valid UTF-8 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object")
            return value

        def _authorized(self) -> bool:
            supplied = self.headers.get("X-Session-Token", "")
            return bool(supplied) and secrets.compare_digest(supplied, state.token)

        def _request_is_local(self) -> bool:
            """Reject DNS-rebinding Host headers before exposing local state."""

            host_header = self.headers.get("Host", "")
            try:
                hostname = urllib.parse.urlsplit(f"//{host_header}").hostname
            except ValueError:
                return False
            return hostname is not None and _is_loopback_host(hostname)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._request_is_local():
                self._error(HTTPStatus.MISDIRECTED_REQUEST, "Host must be a loopback address")
                return
            path, query = self._parse_url()
            if path == "/api/bootstrap":
                self._send_json(state.bootstrap())
                return
            if path in {"/api/jobs", "/api/job", "/api/status"}:
                try:
                    after = max(0, int(query.get("after", ["0"])[0]))
                except ValueError:
                    self._error(HTTPStatus.BAD_REQUEST, "after must be an integer")
                    return
                self._send_json({"ok": True, **state.jobs.snapshot(after)})
                return
            if path == "/api/artifacts":
                files = state.jobs.artifacts()
                self._send_json(
                    {
                        "ok": True,
                        "files": files,
                        "truncated": len(files) >= MAX_ARTIFACTS,
                    }
                )
                return
            if path == "/api/artifact":
                values = query.get("path", [])
                if not values:
                    self._error(HTTPStatus.BAD_REQUEST, "path is required")
                    return
                self._serve_artifact(Path(values[0]))
                return
            self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._request_is_local():
                self._error(HTTPStatus.MISDIRECTED_REQUEST, "Host must be a loopback address")
                return
            if not self._authorized():
                self._error(HTTPStatus.FORBIDDEN, "Missing or invalid X-Session-Token")
                return
            path, _query = self._parse_url()
            try:
                body = self._read_json()
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            if path in {"/api/jobs", "/api/jobs/start"}:
                kind = body.get("kind")
                config = body.get("config")
                if not isinstance(kind, str) or not isinstance(config, Mapping):
                    self._error(HTTPStatus.BAD_REQUEST, "kind and config are required")
                    return
                try:
                    job = state.jobs.start(kind, config)
                except ValueError as exc:
                    self._error(HTTPStatus.BAD_REQUEST, str(exc))
                except RuntimeError as exc:
                    status = HTTPStatus.CONFLICT if "already running" in str(exc) else HTTPStatus.SERVICE_UNAVAILABLE
                    self._error(status, str(exc))
                else:
                    self._send_json({"ok": True, "job": job}, HTTPStatus.ACCEPTED)
                return

            if path == "/api/jobs/cancel":
                try:
                    job = state.jobs.cancel()
                except RuntimeError as exc:
                    self._error(HTTPStatus.CONFLICT, str(exc))
                else:
                    self._send_json({"ok": True, "job": job})
                return

            if path == "/api/pick":
                kind = body.get("kind", "directory")
                title = body.get("title", "")
                initial = body.get("initial", "")
                if not all(isinstance(item, str) for item in (kind, title, initial)):
                    self._error(HTTPStatus.BAD_REQUEST, "picker values must be strings")
                    return
                try:
                    selected = state.pick_path(kind, title, initial)
                except (ValueError, RuntimeError) as exc:
                    self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                else:
                    self._send_json({"ok": True, "path": selected})
                return

            self._error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

        def do_OPTIONS(self) -> None:  # noqa: N802 - reject cross-origin preflights
            if not self._request_is_local():
                self._error(HTTPStatus.MISDIRECTED_REQUEST, "Host must be a loopback address")
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Allow", "GET, POST")
            self._security_headers(api=True)
            self.end_headers()

        def _serve_static(self, request_path: str) -> None:
            aliases = {
                "/": "index.html",
                "/index.html": "index.html",
                "/app.js": "app.js",
                "/styles.css": "styles.css",
            }
            relative = aliases.get(request_path)
            if relative is None and request_path.startswith("/static/"):
                relative = request_path.removeprefix("/static/")
            if relative is None:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                file_path = (WEB_DIR / relative).resolve(strict=True)
            except (OSError, RuntimeError):
                self._error(HTTPStatus.NOT_FOUND, "Static file not found")
                return
            if not file_path.is_file() or not _is_within(file_path, WEB_DIR.resolve(strict=False)):
                self._error(HTTPStatus.NOT_FOUND, "Static file not found")
                return
            self._send_file(file_path, cache="no-cache")

        def _serve_artifact(self, file_path: Path) -> None:
            if not state.jobs.is_allowed_artifact(file_path):
                self._error(HTTPStatus.FORBIDDEN, "Artifact path is outside the selected output directories")
                return
            self._send_file(file_path, cache="no-store", artifact=True)

        def _send_file(self, file_path: Path, *, cache: str, artifact: bool = False) -> None:
            try:
                size = file_path.stat().st_size
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", cache)
                if artifact:
                    encoded_name = urllib.parse.quote(file_path.name, safe="")
                    self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{encoded_name}")
                self._security_headers(api=False, artifact=artifact)
                self.end_headers()
                with file_path.open("rb") as handle:
                    while chunk := handle.read(128 * 1024):
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            except OSError as exc:
                if not self.wfile.closed:
                    self._error(HTTPStatus.NOT_FOUND, f"Could not read file: {exc}")

    return RequestHandler


class LoopbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    state: AppState | None = None,
) -> LoopbackHTTPServer:
    """Create (but do not run) a loopback server; ``port=0`` is supported."""

    if not _is_loopback_host(host):
        raise ValueError("--host must resolve directly to a loopback address (127.0.0.1, ::1, or localhost)")
    if not 0 <= port <= 65_535:
        raise ValueError("--port must be between 0 and 65535")
    app_state = state or AppState()
    server = LoopbackHTTPServer((host, port), create_handler(app_state))
    server.app_state = app_state  # type: ignore[attr-defined]
    return server


def start_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    state: AppState | None = None,
) -> None:
    """Run the local interface until interrupted."""

    server = create_server(host, port, state)
    actual_host, actual_port = server.server_address[:2]
    display_host = f"[{actual_host}]" if ":" in actual_host else actual_host
    url = f"http://{display_host}:{actual_port}/"
    print(f"Symbol Trainer is running at {url}")
    print("Press Ctrl+C to stop it. Files are never uploaded by this server.")
    if open_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping Symbol Trainer…")
    finally:
        server.shutdown()
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local line-symbol trainer interface")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8765, type=int, help="Local port; use 0 to choose a free port")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page automatically")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start_server(args.host, args.port, open_browser=not args.no_browser)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
