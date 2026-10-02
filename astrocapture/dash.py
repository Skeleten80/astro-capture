"""Live web dashboard for a running AstroCapture session.

:class:`AstroDash` runs a full capture session in a background thread —
drivers and :class:`~astrocapture.sequencer.Sequencer` are constructed
exactly the way the CLI builds them (``config.load_plan`` +
``make_mount`` / ``make_camera``) — and serves a mission-control page
plus JSON and Server-Sent Events APIs using only the stdlib
``http.server``.  No web framework required; the only extra dependency
is Pillow, for rendering FITS thumbnails.

Endpoints::

    GET  /              dashboard page
    GET  /api/state     session/target/mount/camera/plan snapshot (JSON)
    GET  /api/thumbnail most recent FITS frame, auto-stretched (PNG)
    GET  /api/log       session.log tail as Server-Sent Events
    POST /api/pause     pause the sequencer
    POST /api/resume    resume the sequencer
    POST /api/abort     abort the sequencer
"""

from __future__ import annotations

import functools
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from astropy.io import fits

from astrocapture import config
from astrocapture.drivers import make_camera, make_mount
from astrocapture.sequencer import SeqState, Sequencer

try:  # Pillow is optional at import time; /api/thumbnail 503s without it.
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is a real dependency
    Image = None  # type: ignore[assignment]

_HTML_PATH = Path(__file__).with_name("dash.html")
_THUMB_MAX_PX = 640
_LOG_BACKLOG_LINES = 200


# -- optional target catalog -------------------------------------------------
def _catalog_lookup(name: str):
    """Look *name* up in astrocapture.catalog if it exists.

    A parallel task may be adding that module; import it defensively and
    try a few plausible lookup spellings.  Returns whatever the catalog
    gave back (dict-like or object) or ``None``.
    """
    try:
        from astrocapture import catalog  # type: ignore[import-not-found]
    except ImportError:
        return None
    for meth in ("lookup", "find", "search", "get", "query"):
        fn = getattr(catalog, meth, None)
        if not callable(fn):
            continue
        try:
            info = fn(name)
        except Exception:  # noqa: BLE001 - catalog must never break the dash
            continue
        if info:
            return info
    return None


def _catalog_field(info, *names):
    """Pull the first matching field out of a dict-like or plain object."""
    if info is None:
        return None
    if isinstance(info, dict):
        for n in names:
            if info.get(n) is not None:
                return info[n]
    for n in names:
        v = getattr(info, n, None)
        if v is not None:
            return v
    return None


class AstroDash:
    """Run a capture session in the background and serve its dashboard.

    The sequencer runs in its own thread; every HTTP handler only reads
    shared state under ``self._lock`` and every handler body is wrapped
    in try/except, so a bad request can never crash the session.
    """

    def __init__(
        self,
        plan: config.Plan,
        mount=None,
        camera=None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.plan = plan
        self.host = host
        self.port = port
        # Same construction path as astrocapture.cli.cmd_run.
        self.mount = (
            mount
            if mount is not None
            else make_mount(plan.mount.driver, **plan.mount.options)
        )
        self.camera = (
            camera
            if camera is not None
            else make_camera(plan.camera.driver, **plan.camera.options)
        )
        self.seq = Sequencer(plan, self.mount, self.camera)
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._seq_thread: threading.Thread | None = None
        self._srv_thread: threading.Thread | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._start_time: float | None = None
        self._catalog_info = _catalog_lookup(plan.target.name)
        # Record completed frames: only paths returned by save_frame are
        # "latest" — a directory scan could catch a FITS mid-write.
        self._latest_frame_path: Path | None = None
        orig_save_frame = self.seq.session.save_frame

        @functools.wraps(orig_save_frame)
        def _recording_save_frame(*args, **kwargs):
            path = orig_save_frame(*args, **kwargs)
            with self._lock:
                self._latest_frame_path = Path(path)
            return path

        self.seq.session.save_frame = _recording_save_frame

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start the session thread and the HTTP server (both background)."""
        self.start_session()
        self.start_server()

    def start_session(self) -> None:
        if self._seq_thread is not None and self._seq_thread.is_alive():
            return
        self._start_time = time.monotonic()
        self._seq_thread = threading.Thread(
            target=self._run_session, name="astro-dash-seq", daemon=True
        )
        self._seq_thread.start()

    def _run_session(self) -> None:
        try:
            self.seq.run()
        except Exception:  # noqa: BLE001 - run() already logs; last resort
            self.seq.log.exception("dashboard session thread crashed")

    def start_server(self) -> None:
        if self._srv_thread is not None and self._srv_thread.is_alive():
            return
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # honour port 0 (ephemeral)
        self._srv_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="astro-dash-http",
            daemon=True,
        )
        self._srv_thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Abort the session (if running) and shut the HTTP server down."""
        self._stopping.set()
        try:
            self.seq.abort()
        except Exception:  # noqa: BLE001 - best effort
            pass
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        for t in (self._srv_thread, self._seq_thread):
            if t is not None:
                t.join(timeout=timeout)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # -- sequencer control (thread-safe; Sequencer documents these as such) --
    def pause(self) -> None:
        with self._lock:
            self.seq.pause()

    def resume(self) -> None:
        with self._lock:
            self.seq.resume()

    def abort(self) -> None:
        with self._lock:
            self.seq.abort()

    # -- state snapshot ------------------------------------------------------
    def state_dict(self) -> dict:
        with self._lock:
            plan = self.plan
            tgt = plan.target
            mount_state, ra, dec = self._mount_snapshot()
            latest = self._latest_fits()
            frames_taken = self.seq.frames_taken
            total_frames = sum(s.count for s in plan.steps)
            return {
                "ok": True,
                "session": {
                    "name": plan.session_name,
                    "dir": str(self.seq.session.dir),
                    "status": self.seq.state.value,
                },
                "target": {
                    "name": tgt.name,
                    "ra_hours": tgt.ra_hours,
                    "dec_deg": tgt.dec_deg,
                    "type": _catalog_field(
                        self._catalog_info, "type", "obj_type", "class", "kind"
                    ),
                    "mag": _catalog_field(
                        self._catalog_info, "mag", "magnitude", "v_mag", "vmag"
                    ),
                },
                "plan": {
                    "frames_taken": frames_taken,
                    "frames_total": total_frames,
                    "current_step": self._current_step(frames_taken),
                    "steps": self._step_progress(frames_taken),
                    "elapsed_s": round(self._elapsed(), 1),
                    "eta_s": round(self._eta(frames_taken, total_frames), 1),
                },
                "mount": {
                    "ra_hours": ra,
                    "dec_deg": dec,
                    "state": mount_state,
                },
                "camera": self._camera_snapshot(),
                "latest_frame": latest.name if latest is not None else None,
            }

    def _mount_snapshot(self) -> tuple[str, float | None, float | None]:
        try:
            state = self.mount.state.value
            ra, dec = self.mount.position
            return state, round(ra, 4), round(dec, 4)
        except Exception:  # noqa: BLE001 - never break /api/state
            return "unknown", None, None

    def _camera_snapshot(self) -> dict:
        cam = self.camera
        exposing = self.seq.state == SeqState.EXPOSING
        started = getattr(cam, "_exposure_start", None)
        in_progress_s = None
        if started is not None:
            exposing = True
            in_progress_s = round(time.monotonic() - started, 1)
        exposure: dict = {"exposure_s": None, "gain": None, "binning": None}
        settings = getattr(cam, "last_settings", None)
        if settings is not None:
            try:
                exptime, gain, binning = settings
                exposure = {
                    "exposure_s": exptime,
                    "gain": gain,
                    "binning": binning,
                }
            except (TypeError, ValueError):
                pass
        try:
            temperature = cam.get_temperature()
        except Exception:  # noqa: BLE001
            temperature = None
        return {
            "state": "exposing" if exposing else "idle",
            "in_progress_s": in_progress_s,
            "last_exposure": exposure,
            "temperature_c": temperature,
        }

    def _current_step(self, frames_taken: int) -> dict | None:
        remaining = frames_taken
        for i, step in enumerate(self.plan.steps):
            if remaining < step.count:
                return {
                    "index": i,
                    "type": step.type,
                    "filter": step.filter,
                    "exposure_s": step.exposure,
                    "frame": remaining + 1,
                    "count": step.count,
                }
            remaining -= step.count
        return None

    def _step_progress(self, frames_taken: int) -> list[dict]:
        out = []
        remaining = frames_taken
        for i, step in enumerate(self.plan.steps):
            done = max(0, min(step.count, remaining))
            remaining -= step.count
            out.append(
                {
                    "index": i,
                    "type": step.type,
                    "filter": step.filter,
                    "exposure_s": step.exposure,
                    "count": step.count,
                    "done": done,
                }
            )
        return out

    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _eta(self, frames_taken: int, total_frames: int) -> float:
        if self.seq.state == SeqState.DONE or total_frames <= frames_taken:
            return 0.0
        if frames_taken > 0 and self._start_time is not None:
            elapsed = self._elapsed()
            if elapsed > 0:
                return max(0.0, elapsed / frames_taken * (total_frames - frames_taken))
        planned = sum(s.count * s.exposure for s in self.plan.steps)
        return max(0.0, planned - self._elapsed())

    def _latest_fits(self) -> Path | None:
        with self._lock:
            recorded = self._latest_frame_path
        if recorded is not None and recorded.is_file():
            return recorded
        # Fallback: scan the session dir (e.g. frames predating the dash).
        try:
            cands = [
                p for p in self.seq.session.dir.rglob("*.fits") if p.is_file()
            ]
        except OSError:
            return None
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    # -- thumbnail -------------------------------------------------------------
    def thumbnail_png(self, max_px: int = _THUMB_MAX_PX) -> bytes | None:
        """Auto-stretched PNG of the most recent FITS frame (None if none)."""
        if Image is None:
            raise RuntimeError("Pillow is not installed; cannot render thumbnail")
        latest = self._latest_fits()
        if latest is None:
            return None
        data = fits.getdata(str(latest)).astype(np.float64)
        # Percentile stretch on a stride for very large sensors.
        sample = data
        if sample.size > 1_000_000:
            stride = int((sample.size / 1_000_000) ** 0.5) + 1
            sample = data[::stride, ::stride]
        lo, hi = (float(v) for v in np.percentile(sample, (1.0, 99.5)))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = float(data.min()), float(data.max())
            if hi <= lo:
                hi = lo + 1.0
        norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        img = Image.fromarray((norm * 255.0).astype(np.uint8))
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # -- HTTP ------------------------------------------------------------------
    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        dash = self

        class DashHandler(BaseHTTPRequestHandler):
            server_version = "AstroDash/0.1"

            def log_message(self, fmt, *args):  # noqa: D102 - keep stderr quiet
                return

            def _send_json(self, code: int, obj: dict) -> None:
                try:
                    body = json.dumps(obj).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _serve_html(self) -> None:
                try:
                    body = _HTML_PATH.read_bytes()
                except OSError:
                    self._send_json(
                        500, {"ok": False, "error": "dashboard page missing"}
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: D102
                try:
                    path = self.path.split("?", 1)[0]
                    if path == "/":
                        self._serve_html()
                    elif path == "/api/state":
                        self._send_json(200, dash.state_dict())
                    elif path == "/api/thumbnail":
                        dash._serve_thumbnail(self)
                    elif path == "/api/log":
                        dash._serve_log_stream(self)
                    else:
                        self._send_json(404, {"ok": False, "error": "not found"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # noqa: BLE001 - never crash a request
                    self._send_json(
                        500,
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    )

            def do_POST(self) -> None:  # noqa: D102
                try:
                    path = self.path.split("?", 1)[0]
                    length = int(self.headers.get("Content-Length") or 0)
                    if length:
                        self.rfile.read(length)
                    if path == "/api/pause":
                        dash.pause()
                    elif path == "/api/resume":
                        dash.resume()
                    elif path == "/api/abort":
                        dash.abort()
                    else:
                        self._send_json(404, {"ok": False, "error": "not found"})
                        return
                    self._send_json(
                        200, {"ok": True, "state": dash.seq.state.value}
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # noqa: BLE001 - never crash a request
                    self._send_json(
                        500,
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    )

        return DashHandler

    def _serve_thumbnail(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            png = self.thumbnail_png()
        except RuntimeError as exc:
            handler._send_json(503, {"ok": False, "error": str(exc)})
            return
        if png is None:
            handler._send_json(404, {"ok": False, "error": "no frames captured yet"})
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "image/png")
        handler.send_header("Content-Length", str(len(png)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(png)

    def _serve_log_stream(self, handler: BaseHTTPRequestHandler) -> None:
        """Stream session.log as Server-Sent Events until disconnect."""
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        log_path = self.seq.session.dir / "session.log"
        pos = 0
        try:
            with open(log_path, "rb") as f:
                data = f.read()
                pos = len(data)
            for line in data.decode("utf-8", "replace").splitlines()[
                -_LOG_BACKLOG_LINES:
            ]:
                _write_sse(handler, line)
            handler.wfile.flush()
        except OSError:
            pass
        try:
            while not self._stopping.is_set():
                try:
                    with open(log_path, "rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                except OSError:
                    chunk = b""
                for line in chunk.decode("utf-8", "replace").splitlines():
                    _write_sse(handler, line)
                handler.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client went away: normal SSE disconnect


def _write_sse(handler: BaseHTTPRequestHandler, line: str) -> None:
    handler.wfile.write(f"data: {line}\n\n".encode("utf-8"))


def serve(config_path: str | Path, port: int = 8765) -> int:
    """Load *config_path*, run the session, serve the dashboard until Ctrl-C.

    Same driver/sequencer construction path as the CLI's ``run`` command.
    """
    plan = config.load_plan(config_path)
    print(config.plan_summary(plan))
    print()
    mount = make_mount(plan.mount.driver, **plan.mount.options)
    camera = make_camera(plan.camera.driver, **plan.camera.options)
    dash = AstroDash(plan, mount=mount, camera=camera, port=port)
    dash.start()
    print(f"AstroCapture dashboard: {dash.url}", flush=True)
    print(f"Session directory: {dash.seq.session.dir}", flush=True)
    print("Press Ctrl-C to stop the dashboard (captured files are kept).", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboard…")
    finally:
        dash.stop()
    return 0
