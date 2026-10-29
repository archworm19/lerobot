"""Wire format shared by the server and the client.

A message is a ZeroMQ multipart message: frame 0 is a UTF-8 JSON header, and any further frames are binary
payloads (encoded images). A request header is `{"id": int, "cmd": str, "args": {...}}`. A reply header is
`{"id": int, "ok": bool, "result": {...}}` or `{"id": int, "ok": false, "error": str}`; `id` echoes the request.
Reply headers can carry `"frames": [{"name": ..., "format": ..., "shape": [...], "dtype": ...}, ...]`, one entry
per binary frame after the header, in order.
"""

import json
from typing import Any

import cv2
import numpy as np

PROTOCOL_VERSION = 2  # 2: capture(after=), head_* commands, head state in get_state
DEFAULT_PORT = 5555


class ServerError(RuntimeError):
    """The server answered a request with an error."""


def encode_image(image: np.ndarray, fmt: str = "png", jpeg_quality: int = 95) -> bytes:
    """Encode an 8-bit RGB image (`fmt` "png" or "jpeg") or a 16-bit single-channel depth image (png only)."""
    if image.dtype == np.uint16:
        if fmt != "png":
            raise ValueError("16-bit images can only be sent as png")
        ok, buf = cv2.imencode(".png", np.squeeze(image))
    elif image.dtype == np.uint8:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if fmt == "png":
            ok, buf = cv2.imencode(".png", bgr)
        elif fmt == "jpeg":
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            raise ValueError(f"unknown image format {fmt!r}")
    else:
        raise ValueError(f"unsupported image dtype {image.dtype}")
    if not ok:
        raise RuntimeError("image encoding failed")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Inverse of `encode_image`: RGB uint8 (H, W, 3), or uint16 (H, W) depth."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode image")
    if img.dtype == np.uint8 and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def pack(header: dict[str, Any], frames: list[bytes] | None = None) -> list[bytes]:
    """The multipart message for a JSON-able `header` followed by binary `frames`."""
    return [json.dumps(header).encode(), *(frames or [])]


def unpack(parts: list[bytes]) -> tuple[dict[str, Any], list[bytes]]:
    """Inverse of `pack`: (header, binary frames)."""
    return json.loads(parts[0].decode()), list(parts[1:])
