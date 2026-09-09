"""Local dashboard server with workbook upload and scoring endpoints."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import Settings
from .upload_pipeline import run_full_production_pipeline, save_uploaded_workbook, workbook_sheets


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
RUN_REPORT_PATH = re.compile(r"^/runs/([A-Za-z0-9_]+)/(business-report|model-analysis)\.xlsx$")
RUN_UPLOAD_PATH = re.compile(r"^/api/uploads/([A-Za-z0-9_]+)/run$")


@dataclass
class UploadState:
    project_root: Path
    uploads: dict[str, Path] = field(default_factory=dict)


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
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as error:
            raise ValueError("A valid Content-Length header is required.") from error
        if length < 1:
            raise ValueError("The request body is empty.")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("The workbook must be 50 MB or smaller.")
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
                workbook_path = self.state.uploads.get(match.group(1))
                if workbook_path is None or not workbook_path.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Upload not found. Upload the workbook again."})
                    return
                payload = json.loads(self._read_body().decode("utf-8"))
                sheet_name = payload.get("sheet_name")
                if not isinstance(sheet_name, str) or not sheet_name:
                    raise ValueError("Choose a worksheet to score.")
                result = run_full_production_pipeline(self.state.project_root, workbook_path, sheet_name)
                result["upload_id"] = match.group(1)
                result["business_report_url"] = f"/runs/{result['run_id']}/business-report.xlsx"
                result["model_analysis_url"] = f"/runs/{result['run_id']}/model-analysis.xlsx"
                self._send_json(HTTPStatus.OK, result)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except FileNotFoundError as error:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except Exception as error:  # Keep the UI response useful while preserving the traceback in the server log.
            print(f"Upload scoring failed: {error}")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Scoring failed. Check the server terminal for details."})


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
