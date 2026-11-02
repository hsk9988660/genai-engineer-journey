import json
import os
import subprocess
import sys

import pytest

from sandbox import Sandbox, SandboxError, ToolOutput, looks_personal, looks_sensitive

SECRET = "sk-testtesttesttest1234"


def make_link(link, target):
    """A directory link: a real symlink where allowed, a Windows junction where symlinks need a privilege. Skips only if neither can be made."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], stdin=subprocess.DEVNULL, capture_output=True)
        if r.returncode == 0:
            return
    pytest.skip("cannot create a directory link on this machine")


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text('def login(u, p):\n    # TODO: hash it\n    return True\n\ndef logout(s):\n    s.clear()\n', encoding="utf8")
    (root / "sub").mkdir()
    (root / "sub" / "util.py").write_text("def clamp(v):\n    return v  # FIXME: bounds\n", encoding="utf8")
    (root / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf8")
    (root / "config.txt").write_text("host = localhost\npassword = hunter2hunter2\n", encoding="utf8")
    (tmp_path / "outside.txt").write_text("OUTSIDE-DATA", encoding="utf8")
    return root


# --- paths ------------------------------------------------------------------------------------------------------------------------------
def test_ordinary_files_resolve_inside_the_root(proj):
    sb = Sandbox(proj)
    assert sb.resolve("auth.py") == proj.resolve() / "auth.py" and sb.resolve("sub/util.py").name == "util.py"


@pytest.mark.parametrize("bad", ["", "   ", None, 5, "/etc/passwd", "C:\\Windows\\win.ini", "C:/Windows/win.ini", "\\\\server\\share\\x", "~/secrets", "../outside.txt", "sub/../../outside.txt",
                                 "sub\\..\\..\\outside.txt", "a\x00b", "a\nb", "x" * 301, "..", "./../outside.txt"])
def test_absolute_parent_home_control_and_oversized_paths_are_refused_before_touching_the_disk(proj, bad):
    with pytest.raises(SandboxError):
        Sandbox(proj).resolve(bad)


def test_a_link_inside_the_project_that_leads_outside_is_refused(proj, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET", encoding="utf8")
    make_link(proj / "lnk", outside)
    with pytest.raises(SandboxError, match="outside"):
        Sandbox(proj).read_text("lnk/secret.txt")


def test_the_root_may_be_given_through_a_link_and_reads_still_work(proj, tmp_path):
    """The lesson never resolves its root, so a linked working directory could make every read look 'outside'."""
    make_link(tmp_path / "via_link", proj)
    sb = Sandbox(tmp_path / "via_link")
    assert "TODO" in sb.read_text("auth.py").text


def test_the_root_must_be_an_existing_directory(tmp_path):
    with pytest.raises(SandboxError):
        Sandbox(tmp_path / "nope")
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(SandboxError):
        Sandbox(tmp_path / "f.txt")


@pytest.mark.parametrize("name", [".env", ".ENV", ".env.local", ".env.staging", "id_rsa", "id_ed25519", "server.pem", "cert.KEY", "store.p12", "credentials.json", ".netrc", ".git/config", ".ssh/id_rsa", ".aws/credentials"])
def test_files_that_commonly_hold_secrets_are_refused_by_name(proj, name):
    p = proj / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf8")
    with pytest.raises(SandboxError, match="secrets|folder"):
        Sandbox(proj).read_text(name)


# --- reading ------------------------------------------------------------------------------------------------------------------------------
def test_a_normal_file_is_returned_whole_and_not_flagged(proj):
    out = Sandbox(proj).read_text("auth.py")
    assert isinstance(out, ToolOutput) and "def login" in out.text and not out.sensitive and not out.truncated


def test_a_file_containing_a_secret_shape_is_returned_but_flagged_sensitive(proj):
    out = Sandbox(proj).read_text("config.txt")
    assert out.sensitive and "hunter2" in out.text


def test_a_big_file_is_cut_at_the_limit_and_says_so(proj):
    (proj / "big.txt").write_text("a" * 50_000, encoding="utf8")
    out = Sandbox(proj, max_read_bytes=1000).read_text("big.txt")
    assert out.truncated and out.text.startswith("a" * 1000) and "[truncated: showing the first 1000 of 50000 bytes]" in out.text and len(out.text) < 1100


def test_a_file_exactly_at_the_limit_is_not_called_truncated(proj):
    (proj / "exact.txt").write_text("b" * 1000, encoding="utf8")
    assert not Sandbox(proj, max_read_bytes=1000).read_text("exact.txt").truncated


def test_a_binary_file_gets_a_placeholder_instead_of_an_exception(proj):
    (proj / "blob.bin").write_bytes(bytes(range(256)) * 10)
    assert Sandbox(proj).read_text("blob.bin").text == "[binary file: contents not shown]"


def test_text_that_is_not_utf8_gets_a_placeholder(proj):
    (proj / "latin.txt").write_bytes("café au lait".encode("latin-1") * 5)
    assert "not UTF-8" in Sandbox(proj).read_text("latin.txt").text


def test_cutting_in_the_middle_of_a_multibyte_character_does_not_break_the_read(proj):
    (proj / "uni.txt").write_text("é" * 2000, encoding="utf8")  # 2 bytes each
    out = Sandbox(proj, max_read_bytes=1001).read_text("uni.txt")
    assert out.truncated and out.text.startswith("é" * 500)


def test_a_directory_or_a_missing_file_is_a_clear_refusal(proj):
    sb = Sandbox(proj)
    for name in ("sub", "missing.py"):
        with pytest.raises(SandboxError, match="no such file"):
            sb.read_text(name)


# --- listing and searching -----------------------------------------------------------------------------------------------------------
def test_listing_shows_ordinary_files_and_hides_secrets_hidden_files_and_other_folders(proj, tmp_path):
    (proj / ".hidden").write_text("x")
    (proj / "key.pem").write_text("x")
    (proj / ".git").mkdir()
    (proj / ".git" / "config").write_text("x")
    names = Sandbox(proj).list_files().text.split(", ")
    assert sorted(names) == ["auth.py", "config.txt", "sub/util.py"]


def test_listing_does_not_follow_a_link_to_a_folder_outside(proj, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    make_link(proj / "lnk", outside)
    assert "secret.txt" not in Sandbox(proj).list_files().text and "TOP-SECRET" not in Sandbox(proj).find_todos().text


def test_the_listing_is_capped(proj):
    for i in range(30):
        (proj / f"f{i:02d}.txt").write_text("x")
    text = Sandbox(proj, max_list=10).list_files().text
    assert "[list cut at 10 files]" in text and text.count(",") == 10


def test_an_empty_project_says_so(tmp_path):
    (tmp_path / "e").mkdir()
    assert Sandbox(tmp_path / "e").list_files().text == "(no files)"


def test_todos_are_found_with_file_and_line_number_and_secret_files_are_not_searched(proj):
    (proj / ".env").write_text("# TODO: rotate this key\n")
    out = Sandbox(proj).find_todos().text.splitlines()
    assert out == ["auth.py:2: # TODO: hash it", "sub/util.py:2: return v  # FIXME: bounds"]


def test_no_todos_says_so_and_results_are_capped(proj):
    (proj / "many.txt").write_text("TODO\n" * 100)
    out = Sandbox(proj).find_todos(max_results=5)
    assert out.truncated and "[cut at 5 results]" in out.text
    (proj / "many.txt").unlink()
    (proj / "auth.py").write_text("x = 1\n")
    (proj / "sub" / "util.py").write_text("x = 1\n")
    assert Sandbox(proj).find_todos().text == "No TODO or FIXME comments found."


def test_a_todo_line_that_holds_a_secret_marks_the_result_sensitive(proj):
    (proj / "notes.txt").write_text(f"TODO: remove {SECRET} from here\n")
    assert Sandbox(proj).find_todos().sensitive


def test_binary_files_are_skipped_by_the_todo_search(proj):
    (proj / "blob.bin").write_bytes(b"\x00TODO\x00" * 50)
    assert "blob.bin" not in Sandbox(proj).find_todos().text


def test_analyze_counts_lines_functions_and_todos_and_inherits_the_refusals(proj):
    info = json.loads(Sandbox(proj).analyze("auth.py").text)
    assert info == {"path": "auth.py", "lines": 6, "functions": 2, "todos": 1, "truncated": False}
    with pytest.raises(SandboxError):
        Sandbox(proj).analyze(".env")


# --- the sensitivity heuristics ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [f"API_KEY={SECRET}", "password = hunter2hunter2", "-----BEGIN RSA PRIVATE KEY-----", "AKIAABCDEFGHIJKLMNOP", "ghp_" + "a" * 24, "Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "secret: s3cr3tvalue"])
def test_common_secret_shapes_are_recognised(text):
    assert looks_sensitive(text)


@pytest.mark.parametrize("text", ["def login(user, password):", "We should rotate credentials soon", "the password field is required", "sk-short", "token = x"])
def test_ordinary_code_and_prose_are_not_flagged(text):
    assert not looks_sensitive(text)


def test_personal_data_shapes_are_recognised():
    for text in ("write to jane.doe@example.com", "SSN 123-45-6789", "card 4111 1111 1111 1111", "this is CONFIDENTIAL"):
        assert looks_personal(text)
    assert not looks_personal("a plain sentence about code")


def test_the_heuristic_misses_secrets_with_no_pattern_which_is_why_it_is_not_the_defense():
    assert not looks_sensitive("the vault unseal phrase is correct horse battery staple")


def test_a_parent_segment_is_refused_even_when_the_path_stays_inside_the_root(proj):
    """Strict by design: 'sub/../auth.py' is inside, but the model has no reason to write '..' and a refusal costs nothing."""
    with pytest.raises(SandboxError, match="parent"):
        Sandbox(proj).resolve("sub/../auth.py")


def test_a_hidden_folder_that_is_not_on_the_deny_list_is_still_not_listed_or_searched(proj):
    (proj / ".cache").mkdir()
    (proj / ".cache" / "notes.txt").write_text("TODO: hidden todo")
    sb = Sandbox(proj)
    assert ".cache" not in sb.list_files().text and "hidden todo" not in sb.find_todos().text


def test_analyze_reports_that_a_big_file_was_cut(proj):
    (proj / "big.py").write_text("def f():\n    pass\n" * 5000)
    info = json.loads(Sandbox(proj, max_read_bytes=1000).analyze("big.py").text)
    assert info["truncated"] is True and info["lines"] < 200


def test_a_folder_whose_real_location_is_outside_the_root_is_not_walked_into(proj, monkeypatch):
    """Stands in for a link we cannot create here: the walk must consult the resolved location of each folder."""
    (proj / "linked").mkdir()
    (proj / "linked" / "x.txt").write_text("TODO: should not be reached")
    real = Sandbox._inside
    monkeypatch.setattr(Sandbox, "_inside", lambda self, p: False if p.name == "linked" else real(self, p))
    sb = Sandbox(proj)
    assert "x.txt" not in sb.list_files().text and "should not be reached" not in sb.find_todos().text


def test_a_file_whose_real_location_is_outside_the_root_is_not_listed_or_searched(proj, monkeypatch):
    (proj / "evil.txt").write_text("TODO: should not be reached")
    real = Sandbox._inside
    monkeypatch.setattr(Sandbox, "_inside", lambda self, p: False if p.name == "evil.txt" else real(self, p))
    sb = Sandbox(proj)
    assert "evil.txt" not in sb.list_files().text and "should not be reached" not in sb.find_todos().text
