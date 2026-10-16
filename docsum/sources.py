"""Получение файлов: локальная папка или публичная папка Google Drive."""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_DRIVE_FOLDER_RE = re.compile(r"drive\.google\.com/.*folders/([A-Za-z0-9_-]+)")


def is_drive_url(source: str) -> bool:
    return bool(_DRIVE_FOLDER_RE.search(source))


def drive_folder_id(url: str) -> str:
    m = _DRIVE_FOLDER_RE.search(url)
    if not m:
        raise ValueError(f"Не похоже на ссылку на папку Google Drive: {url}")
    return m.group(1)


def download_drive_folder(url: str, dest: Path) -> Path:
    """Скачивает публичную папку Drive (доступ «всем, у кого есть ссылка»).

    Google Docs/Sheets/Slides gdown экспортирует в docx/xlsx/pptx.
    Ограничение gdown: не более 50 файлов на уровень папки — для больших папок
    используйте Drive API или скачайте папку архивом и передайте путь.
    """
    try:
        import gdown
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Для Google Drive нужен пакет gdown: pip install gdown") from e

    folder_id = drive_folder_id(url)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Скачиваю папку Google Drive %s → %s", folder_id, dest)
    files = gdown.download_folder(id=folder_id, output=str(dest), quiet=True, resume=True)
    if not files:
        raise RuntimeError(
            "Не удалось скачать файлы. Проверьте, что папка открыта по ссылке, "
            "или скачайте её вручную (ZIP) и передайте локальный путь."
        )
    log.info("Скачано файлов: %d", len(files))
    return dest


def resolve_source(source: str, cache_dir: Path) -> Path:
    if is_drive_url(source):
        return download_drive_folder(source, cache_dir / "drive" / drive_folder_id(source))
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Путь не найден: {path}")
    return path
