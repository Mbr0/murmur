"""Tests for the HTTP hardening both hosted engines share.

Two real loopback servers stand in for "the proxy" and "somewhere else", so a
cross-host redirect is exercised end to end rather than mocked: the assertion
that matters is that the second host never receives a request at all, and so
never sees the credential the first one was given.
"""

import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engines._http import (
    REDIRECT_REFUSED,
    open_no_cross_host_redirect,
    require_https_base_url,
    safe_urlopen,
)
from engines.base import EngineError

LEASE = "Bearer lease-token-do-not-forward"
API_KEY = "sk-do-not-forward"


class _Recorder:
    """What a fake host was asked for, and what it should answer."""

    def __init__(self):
        self.requests = []
        #: ``Location`` returned for ``/redirect``; None answers 200 instead.
        self.redirect_to = None


class _Handler(BaseHTTPRequestHandler):
    recorder: _Recorder = None  # set per-server

    def log_message(self, *args):  # silence the test output
        pass

    def _record(self):
        self.recorder.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "x_api_key": self.headers.get("x-api-key"),
            }
        )

    def _answer(self):
        if self.path == "/redirect" and self.recorder.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.recorder.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._record()
        self._answer()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._record()
        self._answer()


class HttpHardeningTestCase(unittest.TestCase):
    """Starts two independent loopback hosts, ``a`` and ``b``."""

    def setUp(self):
        self.a, self.port_a = self.start_server()
        self.b, self.port_b = self.start_server()

    def start_server(self):
        recorder = _Recorder()
        handler = type("BoundHandler", (_Handler,), {"recorder": recorder})
        server_class = type("BoundServer", (ThreadingHTTPServer,), {"daemon_threads": True})
        server = server_class(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return recorder, port


class RequireHttpsBaseUrlTests(unittest.TestCase):
    def test_an_https_url_is_accepted_and_its_trailing_slash_trimmed(self):
        self.assertEqual(
            require_https_base_url("https://proxy.boske.test/v1/"), "https://proxy.boske.test/v1"
        )

    def test_plain_http_to_a_real_host_is_refused(self):
        for url in ("http://proxy.boske.test", "http://10.0.0.9:8080/v1", "http://example.com"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    require_https_base_url(url)

    def test_loopback_http_is_allowed_so_the_test_servers_work(self):
        for url in (
            "http://127.0.0.1",
            "http://127.0.0.1:8123",
            "http://localhost:8123/v1",
            "http://LOCALHOST:9/v1",
        ):
            with self.subTest(url=url):
                self.assertEqual(require_https_base_url(url), url.rstrip("/"))

    def test_an_empty_relative_or_foreign_scheme_url_is_refused(self):
        for url in ("", "   ", None, "proxy.boske.test", "ftp://proxy.boske.test", "//boske.test"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    require_https_base_url(url)

    def test_the_offending_value_is_named_so_the_error_is_actionable(self):
        with self.assertRaises(ValueError) as caught:
            require_https_base_url("http://boske.test", what="Boske base URL")
        self.assertIn("Boske base URL", str(caught.exception))


class RedirectTests(HttpHardeningTestCase):
    def url(self, port, path="/"):
        return f"http://127.0.0.1:{port}{path}"

    def test_a_cross_host_redirect_is_refused_and_the_credential_never_follows(self):
        self.a.redirect_to = f"http://localhost:{self.port_b}/target"
        request = urllib.request.Request(
            self.url(self.port_a, "/redirect"),
            headers={"Authorization": LEASE, "x-api-key": API_KEY},
            method="GET",
        )

        with self.assertRaises(EngineError) as caught:
            open_no_cross_host_redirect(request, 5.0, urllib.request.urlopen)

        self.assertIn(REDIRECT_REFUSED, str(caught.exception))
        # The second host was never contacted, so it never saw the credential.
        self.assertEqual(self.b.requests, [])
        self.assertEqual(len(self.a.requests), 1)

    def test_a_redirect_to_another_port_on_the_same_name_is_refused_too(self):
        self.a.redirect_to = f"http://127.0.0.1:{self.port_b}/target"
        request = urllib.request.Request(
            self.url(self.port_a, "/redirect"), headers={"Authorization": LEASE}, method="GET"
        )
        with self.assertRaises(EngineError):
            open_no_cross_host_redirect(request, 5.0, urllib.request.urlopen)
        self.assertEqual(self.b.requests, [])

    def test_a_redirect_to_another_scheme_is_refused(self):
        self.a.redirect_to = f"https://127.0.0.1:{self.port_b}/target"
        request = urllib.request.Request(
            self.url(self.port_a, "/redirect"), headers={"Authorization": LEASE}, method="GET"
        )
        with self.assertRaises(EngineError):
            open_no_cross_host_redirect(request, 5.0, urllib.request.urlopen)
        self.assertEqual(self.b.requests, [])

    def test_a_same_host_redirect_is_followed_but_drops_the_credential(self):
        self.a.redirect_to = "/target"
        request = urllib.request.Request(
            self.url(self.port_a, "/redirect"),
            headers={"Authorization": LEASE, "x-api-key": API_KEY},
            method="GET",
        )

        with open_no_cross_host_redirect(request, 5.0, urllib.request.urlopen) as response:
            self.assertEqual(response.status, 200)

        self.assertEqual([call["path"] for call in self.a.requests], ["/redirect", "/target"])
        self.assertEqual(self.a.requests[0]["authorization"], LEASE)
        self.assertIsNone(self.a.requests[1]["authorization"])
        self.assertIsNone(self.a.requests[1]["x_api_key"])

    def test_a_plain_request_still_goes_through_untouched(self):
        request = urllib.request.Request(
            self.url(self.port_a, "/ok"), headers={"Authorization": LEASE}, method="GET"
        )
        with open_no_cross_host_redirect(request, 5.0, urllib.request.urlopen) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.a.requests[0]["authorization"], LEASE)

    def test_safe_urlopen_is_the_default_when_no_opener_is_injected(self):
        self.a.redirect_to = f"http://localhost:{self.port_b}/target"
        request = urllib.request.Request(self.url(self.port_a, "/redirect"), method="GET")
        with self.assertRaises(EngineError):
            open_no_cross_host_redirect(request, 5.0)
        self.assertEqual(self.b.requests, [])

    def test_safe_urlopen_refuses_the_same_redirect(self):
        self.a.redirect_to = f"http://localhost:{self.port_b}/target"
        request = urllib.request.Request(self.url(self.port_a, "/redirect"), method="GET")
        with self.assertRaises(EngineError):
            safe_urlopen(request, timeout=5.0)

    def test_an_injected_test_double_is_called_verbatim(self):
        calls = []

        def fake_open(request, timeout=None):
            calls.append((request.full_url, timeout))
            raise urllib.error.URLError("nope")

        request = urllib.request.Request(self.url(self.port_a, "/ok"), method="GET")
        with self.assertRaises(urllib.error.URLError):
            open_no_cross_host_redirect(request, 3.0, fake_open)
        self.assertEqual(calls, [(self.url(self.port_a, "/ok"), 3.0)])


if __name__ == "__main__":
    unittest.main()
