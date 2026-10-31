"""
Pre-parse format conversion for unsupported document types.

MinerU 3.4+ natively parses .pdf, .docx, .pptx, .xlsx and images.
Legacy Office formats (.doc, .odt, .rtf) are converted to .docx and
legacy presentations (.ppt, .odp) to .pptx via LibreOffice headless,
so MinerU parses them natively (better quality than PDF conversion).
Legacy spreadsheets (.xls, .ods) convert to .xlsx.

Used by both ingestion paths (the command line and Library Upload).
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
import os
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

DOCUMENT_FORMATS = {
    ".pdf", ".xlsx",
    ".docx", ".pptx",  # MinerU 3.4+ parses these natively (better quality than PDF conversion)
}

IMAGE_FORMATS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}

NATIVE_FORMATS = DOCUMENT_FORMATS | IMAGE_FORMATS

CONVERT_MAP: dict[str, str] = {
    # Legacy Office → modern Office so MinerU parses natively (better than PDF)
    ".doc":  "docx",
    ".ppt":  "pptx",
    ".xls":  "xlsx",
    ".odt":  "docx",
    ".odp":  "pptx",
    ".ods":  "xlsx",
    ".rtf":  "docx",
}


ALL_SUPPORTED = NATIVE_FORMATS | set(CONVERT_MAP.keys())

_soffice: str | None = None


def _find_soffice() -> str:
    global _soffice
    if _soffice is not None:
        return _soffice
    path = shutil.which("soffice")
    if path is None:
        venv_bin = Path(sys.executable).parent / "soffice"
        if venv_bin.exists():
            path = str(venv_bin)
    if path is None:
        project_venv = PROJECT_ROOT / ".venv" / "bin" / "soffice"
        if project_venv.exists():
            path = str(project_venv)
    if path is None:
        raise FileNotFoundError(
            "soffice not found — install LibreOffice or symlink "
            "soffice into the venv bin/"
        )
    _soffice = path
    return _soffice


def _convert_one(src: Path, target_fmt: str, out_dir: Path,
                 timeout_s: float = 300.0) -> Path | None:
    """Convert *src* to *target_fmt* in *out_dir* via LibreOffice headless."""
    dest = out_dir / f"{src.stem}.{target_fmt}"
    if dest.exists():
        return dest
    try:
        soffice = _find_soffice()
        result = subprocess.run(
            [soffice, "--headless", "--convert-to", target_fmt,
             "--outdir", str(out_dir), str(src)],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if result.returncode != 0:
            log.warning("soffice failed on %s: %s", src.name,
                        (result.stderr or result.stdout).strip()[:200])
            return None
        if dest.exists():
            return dest
        candidates = list(out_dir.glob(f"{src.stem}.*{target_fmt}"))
        if candidates:
            return candidates[0]
        log.warning("soffice produced no output for %s", src.name)
        return None
    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired:
        log.warning("soffice timed out on %s (%ss)", src.name, timeout_s)
        return None
    except Exception as exc:
        log.warning("conversion error for %s: %s", src.name, exc)
        return None


_SKIP_DIRS = {"__MACOSX", "__pycache__", ".git", "output"}


def discover_files(root: Path, supported: set[str] | None = None,
                   include_images: bool = True) -> list[Path]:
    """Recursively discover supported files, filtering OS artifacts.

    By default only document formats are returned.  Pass
    ``include_images=True`` to also pick up standalone image files.
    """
    if supported is None:
        supported = ALL_SUPPORTED if include_images else (
            DOCUMENT_FORMATS | set(CONVERT_MAP.keys())
        )
    return sorted(
        (p for p in root.rglob("*")
         if p.is_file()
         and p.suffix.lower() in supported
         and not p.name.startswith("._")
         and not (_SKIP_DIRS & set(p.relative_to(root).parts))),
        key=lambda p: p.stat().st_size,
    )


def resolve_stem_collisions(
    files: list[Path],
    root: Path,
    staging_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[Path], dict[Path, str]]:
    """Hash-based dedup + parent-prefix renaming for files sharing a stem.

    MinerU keys output directories on stem alone, so files with the same
    stem — regardless of extension or parent directory — collide.

    * Byte-identical (same MD5): true duplicate, keep first, drop rest.
    * Different content, same stem: keep all with parent-prefixed filenames.
    * Non-colliding files: unchanged.

    If *staging_dir* is given, symlinks with the new names are created
    there and the returned list contains the symlink paths (so downstream
    code picks up the renamed stems automatically).

    Returns ``(clean_files, name_map)`` where *name_map* is
    ``{original_path: unique_flat_filename}`` for every file in
    *clean_files*.
    """
    _log = logger or log

    by_stem: dict[str, list[Path]] = {}
    for p in files:
        by_stem.setdefault(p.stem, []).append(p)

    collisions = {s: ps for s, ps in by_stem.items() if len(ps) > 1}
    if not collisions:
        return files, {p: p.name for p in files}

    drop: set[Path] = set()
    name_map: dict[Path, str] = {}

    for stem, paths in sorted(collisions.items()):
        hashes: list[tuple[Path, str]] = []
        for p in paths:
            h = hashlib.md5()
            with open(p, "rb") as fh:
                while chunk := fh.read(65536):
                    h.update(chunk)
            hashes.append((p, h.hexdigest()))

        seen: dict[str, Path] = {}
        for p, h in hashes:
            if h in seen:
                _log.info("Duplicate (identical content): %s = %s, skipping",
                         p.relative_to(root), seen[h].relative_to(root))
                drop.add(p)
            else:
                seen[h] = p

        unique = [p for p in paths if p not in drop]
        if len(unique) <= 1:
            for p in unique:
                name_map[p] = p.name
            continue

        for p in unique:
            rel = p.relative_to(root)
            parts = rel.parent.parts
            if parts:
                prefix = "__".join(parts).replace(" ", "_")
                new_name = f"{prefix}__{p.name}"
            else:
                new_name = p.name
            name_map[p] = new_name
            _log.info("Stem collision resolved: %s → %s", rel, new_name)

    for p in files:
        if p not in name_map and p not in drop:
            name_map[p] = p.name

    if staging_dir is not None:
        staging_dir.mkdir(parents=True, exist_ok=True)

    clean: list[Path] = []
    for p in files:
        if p in drop:
            continue
        flat_name = name_map[p]
        if staging_dir is not None and flat_name != p.name:
            link = staging_dir / flat_name
            target = p.resolve()
            # Idempotent + atomic: concurrent ingestion runs stage the same links.
            if not (link.is_symlink() and Path(os.readlink(link)) == target):
                tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
                tmp_link.symlink_to(target)
                os.replace(tmp_link, link)
            clean.append(link)
        else:
            clean.append(p)

    n_deduped = len(drop)
    n_renamed = sum(1 for p, n in name_map.items() if n != p.name and p not in drop)
    if n_deduped:
        _log.info("Deduplicated %d identical file(s)", n_deduped)
    if n_renamed:
        _log.info("Renamed %d file(s) to resolve stem collisions", n_renamed)

    return clean, name_map


def ensure_supported(files: list[Path], out_dir: Path) -> list[Path]:
    """Return a list of parser-ready files.

    - Files already in NATIVE_FORMATS pass through unchanged.
    - Files in CONVERT_MAP are converted to the target format under *out_dir*.
    - Unrecognised extensions are dropped with a warning.

    The returned list preserves input order; converted files replace their
    source in the list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    converted = 0
    skipped = 0
    _soffice_warned = False

    from collections import Counter
    native_counts: Counter[str] = Counter()
    convert_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()

    for f in files:
        if f.name.startswith("._"):
            continue
        ext = f.suffix.lower()
        if ext in NATIVE_FORMATS:
            result.append(f)
            native_counts[ext] += 1
        elif ext in CONVERT_MAP:
            target_fmt = CONVERT_MAP[ext]
            try:
                converted_path = _convert_one(f, target_fmt, out_dir)
            except FileNotFoundError:
                converted_path = None
                if not _soffice_warned:
                    log.warning("soffice not found — Office→PDF conversion unavailable")
                    _soffice_warned = True
            if converted_path is not None:
                result.append(converted_path)
                converted += 1
                convert_counts[ext] += 1
            else:
                skipped += 1
                fail_counts[ext] += 1
        else:
            skipped += 1
            skip_counts[ext] += 1
            log.debug("skipping unsupported format: %s", f.name)

    log.info("── file inventory ──")
    log.info("  native (pass-through): %d", sum(native_counts.values()))
    for ext, n in sorted(native_counts.items(), key=lambda x: -x[1]):
        log.info("    %-8s %d", ext, n)
    if convert_counts:
        log.info("  converted: %d", sum(convert_counts.values()))
        for ext, n in sorted(convert_counts.items(), key=lambda x: -x[1]):
            log.info("    %-8s %d  →  .%s", ext, n, CONVERT_MAP[ext])
    if fail_counts:
        log.info("  conversion failed: %d", sum(fail_counts.values()))
        for ext, n in sorted(fail_counts.items(), key=lambda x: -x[1]):
            log.info("    %-8s %d", ext, n)
    if skip_counts:
        log.info("  skipped (unsupported): %d", sum(skip_counts.values()))
        for ext, n in sorted(skip_counts.items(), key=lambda x: -x[1]):
            log.info("    %-8s %d", ext, n)
    log.info("  total parser-ready: %d", len(result))
    return result
