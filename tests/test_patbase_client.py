"""Offline tests for scripts/patbase_client.py (fake HTTP session only)."""
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pbc", ROOT / "scripts" / "patbase_client.py")
pbc = importlib.util.module_from_spec(spec)
import sys; sys.modules["pbc"] = pbc
spec.loader.exec_module(pbc)


def png():
    b = io.BytesIO()
    Image.new("L", (8, 8), 255).save(b, "PNG")
    return b.getvalue()


class Resp:
    def __init__(self, body=b"", status=200, ctype="application/json"):
        self.content = body if isinstance(body, bytes) else ("\ufeff" + json.dumps(body)).encode()
        self.status_code, self.headers = status, {"Content-Type": ctype}


class Cookies(list):
    pass


class Sess:
    def __init__(self, routes):
        self.routes, self.calls, self.cookies = routes, [], Cookies()

    def get(self, url, params=None, timeout=None):
        m = (params or {}).get("method") or "img"
        self.calls.append(m)
        r = self.routes[m]
        if isinstance(r, Exception):
            raise r
        if m == "login" and r.status_code == 200 and "LOGIN_TO_API" in r.content.decode("utf-8-sig"):
            self.cookies.append(type("C", (), {"name": "SessionFarm_GUID"})())
        return r


REC = {"Family": [{"Family": "1", "Publications": [
    {"PN": "CN1", "KD": "A", "CC": "CN", "PD": "2020", "AN": "x", "IMG": 0},
    {"PN": "US2", "KD": "AA", "CC": "US", "PD": "2021", "AN": "y", "IMG": 2}]}]}
BASE = {"version": Resp({"Version": "2.0"}), "getweek": Resp({"Week": "1"})}


def client(extra, **kw):
    s = Sess({**BASE, **extra})
    return pbc.PatBaseClient(session=s, min_interval=0, sleep=lambda x: None, **kw), s


def test_bom_json_parsed_and_no_credentials_reported():
    c, _ = client({})
    r = pbc.check_access(c)
    assert r.status == pbc.NO_CREDENTIALS and r.data["version"] == {"Version": "2.0"}


@pytest.mark.parametrize("msg,status", [
    ("Wrong userid and or password", pbc.BAD_CREDENTIALS),
    ("Access to the API is not allowed for your account", pbc.API_NOT_ENABLED),
])
def test_login_errors_classified(msg, status):
    c, s = client({"login": Resp({"ERROR": msg})}, email="a@b.c", password="pw")
    assert pbc.check_access(c).status == status
    assert s.calls.count("login") == 1          # no retry


def test_network_and_tls_errors():
    import requests
    c, _ = client({"version": requests.exceptions.SSLError("x")})
    assert pbc.check_access(c).status == pbc.TLS
    c, _ = client({"version": requests.exceptions.ConnectTimeout("x")})
    assert pbc.check_access(c).status == pbc.NETWORK


def test_http_403_and_html_are_classified():
    c, _ = client({"version": Resp(b"", 403)})
    assert pbc.check_access(c).status == pbc.FORBIDDEN
    c, _ = client({"version": Resp(b"<html>", 200, "text/html")})
    assert pbc.check_access(c).status == pbc.UNEXPECTED


def test_successful_login_and_getrecord_publications():
    c, _ = client({"login": Resp({"LOGIN_TO_API": "OK"}), "getrecord": Resp(REC)},
                  email="a@b.c", password="pw")
    r = pbc.check_access(c, "1")
    assert r.ok and r.data["getrecord_publications"] == 2


def test_fetch_figures_uses_publication_with_images_and_validates_bytes():
    c, _ = client({"login": Resp({"LOGIN_TO_API": "OK"}), "getrecord": Resp(REC),
                   "img": Resp(png(), ctype="image/png")}, email="a@b.c", password="pw")
    f = c.fetch_figures("1", max_pages=5)
    assert [x.page for x in f] == [1, 2] and f[0].publication_number == "US2"
    assert all(x.status == "retrieved" and x.n_bytes > 0 for x in f)


def test_non_image_response_rejected_and_no_figures_status():
    c, _ = client({"login": Resp({"LOGIN_TO_API": "OK"}), "getrecord": Resp(REC),
                   "img": Resp(b"<html>login</html>", ctype="text/html")},
                  email="a@b.c", password="pw")
    assert c.fetch_figures("1")[0].status == pbc.UNEXPECTED
    c, _ = client({"login": Resp({"LOGIN_TO_API": "OK"}), "getrecord": Resp(REC)},
                  email="a@b.c", password="pw")
    assert c.fetch_figures("1", pn="CN1")[0].status == "no_figures_found"


def test_no_secret_in_error_text():
    c, _ = client({"login": Resp({"ERROR": "Wrong userid and or password"})},
                  email="a@b.c", password="TOPSECRET")
    r = pbc.check_access(c)
    assert "TOPSECRET" not in json.dumps(r.__dict__, default=str)
