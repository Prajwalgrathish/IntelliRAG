"""API error type and error-body helper."""
from __future__ import annotations

from typing import Any, Optional


class ApiError(Exception):
    """An error with an HTTP status, machine-readable code and optional details."""

    def __init__(self, status_code: int, code: str, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_body(code: str, message: str, details: Optional[Any] = None, request_id: Optional[str] = None) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    if request_id:
        body["error"]["request_id"] = request_id
    return body
