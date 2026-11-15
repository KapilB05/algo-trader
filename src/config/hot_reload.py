"""
Hot Reload — Watchdog-based live config reload.

Monitors config/settings.yaml for filesystem changes.
On change: re-validates and hot-swaps the Settings singleton.
No WebSocket drop, no engine restart.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config.settings_loader import Settings, reload_settings

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path("config/settings.yaml")


class ConfigReloadHandler(FileSystemEventHandler):
    """Watchdog handler that triggers config reload on file changes."""

    def __init__(
        self,
        config_path: Path,
        on_reload: Optional[Callable[[Settings], None]] = None,
    ) -> None:
        self._config_path = config_path
        self._on_reload = on_reload
        self._last_reload: float = 0
        self._debounce_seconds = 1.0  # Prevent rapid reloads

    def on_modified(self, event: Any) -> None:
        if not isinstance(event, FileModifiedEvent):
            return

        # Only react to our settings file
        if Path(str(event.src_path)).name != self._config_path.name:
            return

        # Debounce
        now = time.time()
        if now - self._last_reload < self._debounce_seconds:
            return
        self._last_reload = now

        logger.info("Config file changed — reloading settings...")
        try:
            new_settings = reload_settings(self._config_path)
            logger.info("✅ Settings hot-reloaded successfully.")

            if self._on_reload:
                self._on_reload(new_settings)

        except Exception as exc:
            logger.error("❌ Hot-reload failed: %s — keeping old settings.", exc)


class HotReloadWatcher:
    """
    Watches config files and triggers hot-reload on changes.

    Usage:
        watcher = HotReloadWatcher(on_reload=callback)
        watcher.start()
        # ... later ...
        watcher.stop()
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        on_reload: Optional[Callable[[Settings], None]] = None,
    ) -> None:
        self._config_path = config_path or SETTINGS_FILE
        self._handler = ConfigReloadHandler(self._config_path, on_reload)
        self._observer = Observer()
        self._running = False

    def start(self) -> None:
        """Start watching for config changes."""
        watch_dir = str(self._config_path.parent)
        self._observer.schedule(self._handler, watch_dir, recursive=False)
        self._observer.start()
        self._running = True
        logger.info("Hot-reload watcher started for %s", self._config_path)

    def stop(self) -> None:
        """Stop the watcher."""
        if self._running:
            self._observer.stop()
            self._observer.join()
            self._running = False
            logger.info("Hot-reload watcher stopped.")

    @property
    def is_running(self) -> bool:
        return self._running
