"""Write-path tests for the memory editing API (backend/api/memory.py).

Memory editing mutates the user's MEMORY.md / USER.md via fcntl locking +
atomic writes, so a bug here can corrupt real agent data. These cover the
risky surface: the ``\n§\n`` entry-delimiter round-trip contract, substring
matching, atomic writes leaving no temp files, and the validation guards.

The endpoint functions are plain ``def`` (FastAPI auto-threads them), so they
are exercised directly. ``default_hermes_dir()`` reads ``HERMES_HOME`` at call
time, so pointing it at a tmp dir is just an env var.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.api.memory import (
    ENTRY_DELIMITER,
    AddBody,
    DeleteBody,
    EditBody,
    _read_entries,
    add_entry,
    delete_entry,
    edit_entry,
)


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _memory_file(home: Path, target: str = "memory") -> Path:
    name = "USER.md" if target == "user" else "MEMORY.md"
    return home / "memories" / name


def test_add_creates_file_with_single_entry(hermes_home: Path) -> None:
    result = add_entry(AddBody(target="memory", content="first fact"))
    assert result == {"ok": True, "entry_count": 1}

    path = _memory_file(hermes_home)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "first fact\n"


def test_add_appends_with_delimiter_and_round_trips(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="alpha"))
    add_entry(AddBody(target="memory", content="beta"))

    path = _memory_file(hermes_home)
    assert path.read_text(encoding="utf-8") == "alpha" + ENTRY_DELIMITER + "beta" + "\n"
    # the on-disk format parses back to the original entries
    assert _read_entries("memory") == ["alpha", "beta"]


def test_add_rejects_empty_content(hermes_home: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        add_entry(AddBody(target="memory", content="   "))
    assert exc.value.status_code == 400


def test_add_rejects_duplicate(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="dup"))
    with pytest.raises(HTTPException) as exc:
        add_entry(AddBody(target="memory", content="dup"))
    assert exc.value.status_code == 409
    # the duplicate did not double-write
    assert _read_entries("memory") == ["dup"]


def test_memory_and_user_targets_are_isolated(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="in memory"))
    add_entry(AddBody(target="user", content="in user"))

    assert _read_entries("memory") == ["in memory"]
    assert _read_entries("user") == ["in user"]
    assert _memory_file(hermes_home, "memory").read_text(encoding="utf-8") == "in memory\n"
    assert _memory_file(hermes_home, "user").read_text(encoding="utf-8") == "in user\n"


def test_edit_replaces_matched_entry_preserving_others(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="keep one"))
    add_entry(AddBody(target="memory", content="change me"))
    add_entry(AddBody(target="memory", content="keep two"))

    result = edit_entry(EditBody(target="memory", old_text="change", content="changed!"))
    assert result["ok"] is True
    assert _read_entries("memory") == ["keep one", "changed!", "keep two"]


def test_edit_no_match_returns_404(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="something"))
    with pytest.raises(HTTPException) as exc:
        edit_entry(EditBody(target="memory", old_text="absent", content="x"))
    assert exc.value.status_code == 404
    assert _read_entries("memory") == ["something"]


def test_edit_ambiguous_match_is_rejected_without_writing(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="shared token A"))
    add_entry(AddBody(target="memory", content="shared token B"))

    with pytest.raises(HTTPException) as exc:
        edit_entry(EditBody(target="memory", old_text="shared token", content="x"))
    assert exc.value.status_code == 409
    # both entries are intact — an ambiguous edit must not clobber data
    assert _read_entries("memory") == ["shared token A", "shared token B"]


def test_edit_rejects_empty_content(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="x"))
    with pytest.raises(HTTPException) as exc:
        edit_entry(EditBody(target="memory", old_text="x", content="  "))
    assert exc.value.status_code == 400


def test_delete_removes_matched_entry_preserving_others(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="alpha"))
    add_entry(AddBody(target="memory", content="bravo"))
    add_entry(AddBody(target="memory", content="charlie"))

    result = delete_entry(DeleteBody(target="memory", old_text="bravo"))
    assert result == {"ok": True, "entry_count": 2}
    assert _read_entries("memory") == ["alpha", "charlie"]


def test_delete_last_entry_empties_file(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="only"))
    delete_entry(DeleteBody(target="memory", old_text="only"))

    assert _read_entries("memory") == []
    assert _memory_file(hermes_home).read_text(encoding="utf-8") == ""


def test_delete_no_match_returns_404(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="present"))
    with pytest.raises(HTTPException) as exc:
        delete_entry(DeleteBody(target="memory", old_text="absent"))
    assert exc.value.status_code == 404
    assert _read_entries("memory") == ["present"]


def test_delete_ambiguous_match_is_rejected_without_writing(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="dupe x"))
    add_entry(AddBody(target="memory", content="dupe y"))

    with pytest.raises(HTTPException) as exc:
        delete_entry(DeleteBody(target="memory", old_text="dupe"))
    assert exc.value.status_code == 409
    assert _read_entries("memory") == ["dupe x", "dupe y"]


def test_atomic_writes_leave_no_temp_files(hermes_home: Path) -> None:
    add_entry(AddBody(target="memory", content="a"))
    edit_entry(EditBody(target="memory", old_text="a", content="b"))
    delete_entry(DeleteBody(target="memory", old_text="b"))

    leftovers = list((hermes_home / "memories").glob("*.tmp"))
    assert leftovers == []


def test_memory_routes_are_registered() -> None:
    from backend.main import app

    methods = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }
    assert ("GET", "/api/memory") in methods
    assert ("POST", "/api/memory") in methods
    assert ("PUT", "/api/memory") in methods
    assert ("DELETE", "/api/memory") in methods
