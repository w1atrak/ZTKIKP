"""Homomorphic linear score and the HTTP API. The secret key is not here."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from phe.paillier import EncryptedNumber

from ppspam import config, crypto, features, protocol, quantize

logger = logging.getLogger("ppspam.server")


class Server:
    """Computes ``Enc(b_q' + x_q · w_q)`` from a dense encrypted vector."""

    def __init__(
        self,
        weights: dict | None = None,
        weights_path: Path | None = None,
        model_version: str | None = None,
    ):
        if weights is None:
            path = (
                quantize.SERVER_WEIGHTS_Q_JSON if weights_path is None else weights_path
            )
            weights = json.loads(Path(path).read_text())
        self.w_q = [int(value) for value in weights["w_q"]]
        self.b_q_prime = int(weights["b_q_prime"])
        self.d = int(weights.get("d", len(self.w_q)))
        if self.d != len(self.w_q):
            raise ValueError(f"d={self.d} does not match {len(self.w_q)} weights")
        self.model_version = str(
            weights.get("model_version", model_version or config.MODEL_VERSION)
        )

    def score(self, public_key, enc_vector) -> EncryptedNumber:
        """``Enc(b_q') + Σ enc_i * w_q[i]``. A zero weight skips the multiplication."""
        self._validate(public_key, enc_vector)
        acc = public_key.encrypt(self.b_q_prime)
        for cipher, weight in zip(enc_vector, self.w_q, strict=True):
            if weight == 0:
                continue
            acc = acc + cipher * int(weight)
        return acc

    def score_request(self, payload: dict) -> EncryptedNumber:
        """Deserialize an in-process client payload and return the encrypted score."""
        crypto.assert_public_payload(payload)
        if "public_key" not in payload or "ciphertexts" not in payload:
            raise ValueError("payload needs public_key and ciphertexts")
        public_key = crypto.load_public_key(payload["public_key"])
        enc_vector = [
            crypto.ciphertext_from_pair(public_key, pair)
            for pair in payload["ciphertexts"]
        ]
        return self.score(public_key, enc_vector)

    def score_protocol(self, body: dict) -> dict:
        """``POST /score`` body → ``{"result", "exponent"}`` (wire format, B5)."""
        n, ciphertexts, model_version = protocol.parse_score_request(body)
        if model_version != self.model_version:
            raise protocol.VersionMismatch(
                f"model_version {model_version!r} != {self.model_version!r}"
            )
        if len(ciphertexts) != self.d:
            raise protocol.ProtocolError(
                f"ciphertexts has length {len(ciphertexts)}, expected {self.d}"
            )
        public_key = crypto.load_public_key({"n": n})
        enc_vector = [
            crypto.ciphertext_from_pair(public_key, (ct, 0)) for ct in ciphertexts
        ]
        result = self.score(public_key, enc_vector)
        ct, exp = crypto.ciphertext_to_pair(result)
        return protocol.score_response(ct, exp)

    def _validate(self, public_key, enc_vector) -> None:
        # Two n_length/2 primes can multiply to n_length-1 bits.
        if public_key.n.bit_length() < config.KEY_SIZE_DEV - 1:
            raise ValueError(
                f"modulus n has {public_key.n.bit_length()} bits; "
                f"need at least {config.KEY_SIZE_DEV - 1}"
            )
        if len(enc_vector) != self.d:
            raise ValueError(
                f"encrypted vector has length {len(enc_vector)}, expected {self.d}"
            )
        for item in enc_vector:
            if not isinstance(item, EncryptedNumber):
                raise TypeError("encrypted vector entries must be EncryptedNumber")
            if item.public_key != public_key:
                raise ValueError("ciphertext was encrypted under a different key")
            raw = item.ciphertext(be_secure=False)
            if not 0 <= raw < public_key.nsquare:
                raise ValueError("ciphertext is outside 0 .. n²-1")


def create_app(
    weights_path: Path | None = None,
    public_model_path: Path | None = None,
    model_version: str | None = None,
) -> FastAPI:
    """FastAPI app: ``GET /public-model``, ``POST /score``. No private key."""
    server = Server(weights_path=weights_path, model_version=model_version)
    public_path = (
        features.PUBLIC_MODEL_JSON if public_model_path is None else public_model_path
    )
    public_raw = json.loads(Path(public_path).read_text())
    if int(public_raw["d"]) != server.d:
        raise ValueError(
            f"public model d={public_raw['d']} != server weights d={server.d}"
        )
    public_message = protocol.public_model_message(
        public_raw, model_version=server.model_version
    )

    app = FastAPI(title="ppspam", version=server.model_version)
    app.state.server = server
    app.state.public_message = public_message

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                return JSONResponse({"detail": "invalid content-length"}, status_code=400)
            if size > protocol.MAX_SCORE_BODY_BYTES:
                logger.warning(
                    "rejected oversized request path=%s content_length=%s",
                    request.url.path,
                    size,
                )
                return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        logger.info("validation error: %s", exc.errors())
        return JSONResponse({"detail": "invalid request"}, status_code=422)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": server.model_version, "d": server.d}

    @app.get("/public-model")
    def get_public_model():
        return public_message

    @app.post("/score")
    async def post_score(request: Request):
        raw = await request.body()
        if len(raw) > protocol.MAX_SCORE_BODY_BYTES:
            logger.warning("rejected oversized body bytes=%s", len(raw))
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        started = time.perf_counter()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"detail": "body must be JSON"}, status_code=400)
        try:
            # Never log ciphertexts or the modulus digits — only shape metadata.
            n_cts = (
                len(body["ciphertexts"])
                if isinstance(body, dict) and isinstance(body.get("ciphertexts"), list)
                else None
            )
            version = body.get("model_version") if isinstance(body, dict) else None
            response = server.score_protocol(body)
        except protocol.VersionMismatch as exc:
            logger.info("version mismatch model_version=%r", version)
            return JSONResponse({"detail": str(exc)}, status_code=409)
        except protocol.ProtocolError as exc:
            logger.info("protocol error d=%s detail=%s", n_cts, exc)
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except ValueError as exc:
            logger.info("score rejected d=%s detail=%s", n_cts, exc)
            return JSONResponse({"detail": str(exc)}, status_code=400)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "score ok d=%s model_version=%s elapsed_ms=%.1f",
            n_cts,
            version,
            elapsed_ms,
        )
        return response

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
