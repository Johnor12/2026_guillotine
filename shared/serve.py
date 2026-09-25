#!/usr/bin/env python3
"""Serve the dashboards locally at http://127.0.0.1:8123.

Usage:
    uv run -m shared.serve
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .paths import OUT

ADDRESS = ("127.0.0.1", 8123)


def main() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=OUT)
    with ThreadingHTTPServer(ADDRESS, handler) as server:
        print(
            f"Serving the season desk at http://{ADDRESS[0]}:{ADDRESS[1]}/, "
            "the draft board at /draft.html and the source investigator at /sources.html"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
