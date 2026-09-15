"""Local dashboard server with workbook upload and scoring endpoints."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import Settings
from .upload_pipeline import run_full_production_pipeline, save_uploaded_workbook, workbook_sheets


MAX_UPLOAD_BYTES = 65 * 1024 * 1024
RUN_REPORT_PATH = re.compile(r"^/runs/([A-Za-z0-9_]+)/(business-report|model-analysis)\.xlsx$")
RUN_UPLOAD_PATH = re.compile(r"^/api/uploads/([A-Za-z0-9_]+)/run$")
RUN_STATUS_PATH = re.compile(r"^/api/runs/([A-Za-z0-9_]+)$")


@dataclass
class UploadState:
    project_root: Path
    uploads: dict[str, Path] = field(default_factory=dict)
    jobs: dict[str, dict] = field(default_factory=dict)
    jobs_lock: threading.Lock = field(default_factory=threading.Lock)


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    state: UploadState

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(format % args)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            # The browser may leave after a long-running request fails.
            pass

    def _read_body(self) -> bytes:
        values = self.headers.get_all("Content-Length") or []
        try:
            lengths = [int(value) for value in values]
        except ValueError as error:
            raise ValueError("A valid Content-Length header is required.") from error
        if any(length > MAX_UPLOAD_BYTES for length in lengths):
            raise ValueError("The workbook must be 65 MB or smaller.")
        if len(set(lengths)) > 1:
            raise ValueError("Conflicting Content-Length headers are not allowed.")
        length = lengths[-1] if lengths else 0
        if length < 1:
            raise ValueError("The request body is empty.")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("The workbook must be 65 MB or smaller.")
        return self.rfile.read(length)

    def _serve_report(self, run_id: str, report: str) -> None:
        filename = {
            "business-report": "Business Report.xlsx",
            "model-analysis": "Model Analysis.xlsx",
        }[report]
        path = self.state.project_root / "artifacts" / "uploads" / run_id / filename
        if not path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Report not found."})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(path.read_bytes())

    def _job_response(self, job_id: str) -> dict | None:
        with self.state.jobs_lock:
            job = self.state.jobs.get(job_id)
            return dict(job) if job else None

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/dashboard-data":
            dashboard_path = self.state.project_root / "artifacts" / "dashboard" / "dashboard_data.json"
            if not dashboard_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Dashboard data has not been exported yet."})
                return
            self._send_json(HTTPStatus.OK, json.loads(dashboard_path.read_text(encoding="utf-8")))
            return
        match = RUN_REPORT_PATH.match(path)
        if match:
            self._serve_report(match.group(1), match.group(2))
            return
        match = RUN_STATUS_PATH.match(path)
        if match:
            job = self._job_response(match.group(1))
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Run not found."})
            else:
                self._send_json(HTTPStatus.OK, job)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/uploads":
                filename = unquote(self.headers.get("X-File-Name", ""))
                upload_id, workbook_path = save_uploaded_workbook(
                    self.state.project_root,
                    filename,
                    self._read_body(),
                )
                sheets = workbook_sheets(workbook_path)
                if not sheets:
                    raise ValueError("The workbook does not contain any worksheets.")
                self.state.uploads[upload_id] = workbook_path
                self._send_json(HTTPStatus.CREATED, {"upload_id": upload_id, "sheets": sheets})
                return

            match = RUN_UPLOAD_PATH.match(path)
            if match:
                upload_id = match.group(1)
                workbook_path = self.state.uploads.get(upload_id)
                if workbook_path is None:
                    candidates = (self.state.project_root / "data" / "uploads").glob(f"{upload_id}_*")
                    workbook_path = next(candidates, None)
                if workbook_path is None or not workbook_path.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Upload not found. Upload the workbook again."})
                    return
                payload = json.loads(self._read_body().decode("utf-8"))
                sheet_name = payload.get("sheet_name")
                if not isinstance(sheet_name, str) or not sheet_name:
                    raise ValueError("Choose a worksheet to score.")
                job_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
                job = {
                    "job_id": job_id,
                    "status": "queued",
                    "progress": 0,
                    "message": "Run queued.",
                    "upload_id": upload_id,
                    "upload_name": workbook_path.name,
                    "sheet_name": sheet_name,
                }
                with self.state.jobs_lock:
                    self.state.jobs[job_id] = job

                threading.Thread(
                    target=self._run_job,
                    args=(job_id, upload_id, workbook_path, sheet_name),
                    daemon=True,
                ).start()
                self._send_json(HTTPStatus.ACCEPTED, job)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except FileNotFoundError as error:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except Exception as error:  # Keep the UI response useful while preserving the traceback in the server log.
            print(f"Upload scoring failed: {error}")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Scoring failed. Check the server terminal for details."})

    def _run_job(self, job_id: str, upload_id: str, workbook_path: Path, sheet_name: str) -> None:
        def update(progress: float, message: str) -> None:
            with self.state.jobs_lock:
                job = self.state.jobs[job_id]
                job.update(status="running", progress=progress, message=message)

        try:
            update(0, "Starting production model.")
            result = run_full_production_pipeline(
                self.state.project_root,
                workbook_path,
                sheet_name,
                progress_callback=update,
            )
            result["upload_id"] = upload_id
            result["business_report_url"] = f"/runs/{result['run_id']}/business-report.xlsx"
            result["model_analysis_url"] = f"/runs/{result['run_id']}/model-analysis.xlsx"
            with self.state.jobs_lock:
                self.state.jobs[job_id].update(status="completed", progress=1, message="Production run complete.", result=result)
        except Exception as error:
            print(f"Upload scoring failed: {error}")
            with self.state.jobs_lock:
                self.state.jobs[job_id].update(status="failed", message=str(error), error=str(error))


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    settings = Settings()
    state = UploadState(settings.project_root)
    dashboard_dir = settings.project_root / "artifacts" / "dashboard"

    class Handler(DashboardRequestHandler):
        pass

    Handler.state = state
    return ThreadingHTTPServer((host, port), lambda *args, **kwargs: Handler(*args, directory=str(dashboard_dir), **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local game-performance dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"Dashboard available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
