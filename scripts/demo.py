#!/usr/bin/env python3
"""Interactive SMS spam demo: type a message, get ham/spam + timings (B5).

Local mode (default) scores in-process. Network mode talks to a running API::

    Terminal 1:  make serve
    Terminal 2:  make demo ARGS='--url http://127.0.0.1:8000'
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ppspam import config, crypto  # noqa: E402
from ppspam.client import Client  # noqa: E402
from ppspam.server import Server  # noqa: E402


def _format_timings(timings: dict[str, float]) -> str:
    parts = [f"{key.replace('_s', '')}={value * 1000:.1f} ms" for key, value in timings.items()]
    return ", ".join(parts)


def _score_local(client: Client, server: Server, text: str) -> tuple[int, int]:
    payload = client.prepare(text)
    started = time.perf_counter()
    encrypted = server.score_request(payload)
    client.timings = {
        **client.timings,
        "server_s": time.perf_counter() - started,
    }
    label, score = client.decide(crypto.ciphertext_to_pair(encrypted))
    client.timing_log[-1] = {**client.timing_log[-1], **client.timings}
    return label, score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help="API base URL (default: local in-process scoring)",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=config.KEY_SIZE_DEV,
        help=f"Paillier modulus bits (default {config.KEY_SIZE_DEV} for a snappy demo)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Score a single SMS and exit (non-interactive)",
    )
    args = parser.parse_args(argv)

    print(f"Generating {args.key_size}-bit Paillier keypair…", flush=True)
    if args.url:
        client = Client.from_server(args.url, n_length=args.key_size)
        server = None
        print(f"Using remote server at {args.url} (d={client.featurizer.d})", flush=True)
    else:
        client = Client(n_length=args.key_size)
        server = Server()
        print(f"Local in-process scoring (d={client.featurizer.d})", flush=True)

    def run_once(text: str) -> None:
        text = text.strip()
        if not text:
            print("(empty message skipped)")
            return
        if server is None:
            label, score = client.score_remote(text)
        else:
            label, score = _score_local(client, server, text)
        name = "spam" if label == 1 else "ham"
        print(f"→ {name}  (s = {score})")
        print(f"  timings: {_format_timings(client.timings)}")

    if args.text is not None:
        run_once(args.text)
        return 0

    print("Type an SMS and press Enter (Ctrl-D / Ctrl-C to quit).")
    print("Examples:  'WINNER!! claim your prize now'  /  'see you at 7'")
    try:
        while True:
            try:
                line = input("sms> ")
            except EOFError:
                print()
                break
            run_once(line)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
