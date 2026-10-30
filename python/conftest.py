from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path

if os.name == "nt":
    _orig_mkdir = os.mkdir

    def _safe_windows_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        # En Python 3.13+ sobre Windows, mode=0o700 aplica una DACL restrictiva
        # que provoca PermissionError (WinError 5) al listar o limpiar tmp_path en pytest.
        effective_mode = 0o777 if mode == 0o700 else mode
        if dir_fd is not None:
            _orig_mkdir(path, effective_mode, dir_fd=dir_fd)
        else:
            _orig_mkdir(path, effective_mode)

    os.mkdir = _safe_windows_mkdir  # type: ignore[assignment]

    if not os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        default_root = Path(tempfile.gettempdir()).resolve() / f"pytest-of-{user}"
        try:
            if default_root.exists():
                list(os.scandir(default_root))
        except PermissionError:
            fallback_root = Path(tempfile.gettempdir()).resolve() / "pytest_pedemonte_tmp"
            fallback_root.mkdir(parents=True, exist_ok=True)
            os.environ["PYTEST_DEBUG_TEMPROOT"] = str(fallback_root)

