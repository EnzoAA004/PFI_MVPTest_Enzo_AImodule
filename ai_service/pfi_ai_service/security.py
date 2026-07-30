from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {
    "path",
    "sourcePath",
    "inputPath",
    "input_path",
    "imagePath",
    "maskPath",
    "confidencePath",
    "overlayPath",
    "maskPreviewPath",
    "overlay_path",
    "outputFiles",
    "outputPath",
    "checkpointPath",
    "modelPath",
    "token",
    "accessToken",
    "refreshToken",
    "authorization",
    "password",
    "secret",
}

SAFE_PATH_KEYS = {"path", "method"}
PATH_LIKE = re.compile(
    r"([A-Za-z]:[\\/]|/tmp\b|/var\b|/app\b|/content\b|/models\b|/outputs\b|\\\\|localhost|host\.docker\.internal|cloudflare)",
    re.IGNORECASE,
)
TOKEN_LIKE = re.compile(r"(bearer\s+[A-Za-z0-9._~+/=-]+|eyJ[A-Za-z0-9._~+/=-]+)", re.IGNORECASE)


def sanitize_public_payload(value: Any, *, keep_error_path: bool = False) -> Any:
    if isinstance(value, Path):
        return None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _drop_key(text_key, keep_error_path=keep_error_path):
                continue
            clean = sanitize_public_payload(item, keep_error_path=keep_error_path)
            if clean is not None:
                sanitized[text_key] = clean
        return sanitized
    if isinstance(value, list):
        return [
            clean
            for item in value
            if (clean := sanitize_public_payload(item, keep_error_path=keep_error_path)) is not None
        ]
    if isinstance(value, tuple):
        return [
            clean
            for item in value
            if (clean := sanitize_public_payload(item, keep_error_path=keep_error_path)) is not None
        ]
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value


def sanitize_public_text(value: str) -> str:
    cleaned = TOKEN_LIKE.sub("[redacted-token]", value)
    if PATH_LIKE.search(cleaned):
        return "[redacted-internal-detail]"
    return cleaned


def safe_exception_type(exc: Exception) -> str:
    return type(exc).__name__


def _drop_key(key: str, *, keep_error_path: bool) -> bool:
    if keep_error_path and key in SAFE_PATH_KEYS:
        return False
    normalized = key.lower()
    return key in SENSITIVE_KEYS or normalized.endswith("token") or "secret" in normalized or "password" in normalized
