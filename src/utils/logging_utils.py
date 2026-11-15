"""
Logging utilities — Secret Redaction & Robust Output.

The Angel One SmartAPI SDK logs full request headers (including the JWT
`Authorization` token and `X-PrivateKey`) on every network failure. This
module installs a logging filter that scrubs those secrets from every
emitted record, regardless of which handler writes them.

Also provides a UTF-8-safe stdout/stderr reconfigure helper so that the
rupee glyph and emoji used across log messages cannot crash a handler with
UnicodeEncodeError on non-UTF-8 consoles/pipes.
"""

from __future__ import annotations

import logging
import re
import sys

_JWT_RE = re.compile(r"Bearer [A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)('(?:x-privatekey|x-sessiontoken|x-usertoken|authorization|refresh-token)'\s*:\s*')[^']*(')"
)


class SecretRedactionFilter(logging.Filter):
    """Redacts credentials tokens from every log record message."""

    def _redact(self, text: str) -> str:
        text = _JWT_RE.sub("<REDACTED_JWT>", text)
        # Quoted header mappings inside SDK error dumps, e.g. {'X-PrivateKey': 'AODOZmpW'}
        text = _SENSITIVE_VALUE_RE.sub(r"\1<REDACTED>\2", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = self._redact(msg)
        if redacted != msg:
            record.msg = redacted
            record.args = None
        return True


def _install_filter_on_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.addFilter(SecretRedactionFilter())


def install_secret_redaction() -> None:
    """Attach the secret-redaction filter to the loggers that emit credentials.

    The SmartAPI SDK logs under its own logger (``smartConnect``/``SmartApi``)
    which propagates to the root logger. Attaching here survives uvicorn's
    ``dictConfig`` because those named loggers are not reconfigured by it.
    """
    for name in ("smartConnect", "SmartApi", "SmartConnect"):
        _install_filter_on_logger(name)
    _install_filter_on_logger("root")


def make_utf8_stdio() -> None:
    """Reconfigure stdout/stderr so emoji and INR glyphs never crash a handler."""
    for stream_name in ("stdout", "stderr"):
        try:
            stream = getattr(sys, stream_name)
            if stream is not None:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass