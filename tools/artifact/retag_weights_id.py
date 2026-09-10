#!/usr/bin/env python3
"""Rewrite a .ninfer container's weights_id in place, without moving the payload.

Why this exists: the QUASAR artifact is published claiming `qwen3.8-27b/nvfp4`,
the same identity the official mixed-precision artifact already uses. Our engine
registers it as `nvfp4-quasar` instead, so that resolve_weights() stays a total
function of the identity rather than sniffing stored tensor formats to tell three
artifacts apart. Retagging the download is what reconciles the two.

The container is:

    magic(8) | json_bytes(8 LE) | JSON directory | pad to 4096 | payload

and every object offset in the directory is relative to payload_start, which is
align_up(16 + json_bytes, 4096). So as long as the grown JSON still lands inside
the existing alignment padding, payload_start does not move and neither does a
single byte of the ~20 GB payload. This refuses to run when that is not true,
rather than silently producing a container whose offsets are all wrong.

Usage:  retag_weights_id.py <artifact.ninfer> <new-weights-id>
"""

from __future__ import annotations

import json
import struct
import sys

MAGIC = b"NINFER\x00\x02"
PREFIX_BYTES = 16
PAYLOAD_ALIGNMENT = 4096


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: retag_weights_id.py <artifact.ninfer> <new-weights-id>",
              file=sys.stderr)
        return 2
    path, new_id = argv[1], argv[2]

    with open(path, "r+b") as handle:
        prefix = handle.read(PREFIX_BYTES)
        if prefix[:8] != MAGIC:
            print(f"{path}: not an NInfer v2 container", file=sys.stderr)
            return 1
        (json_bytes,) = struct.unpack("<Q", prefix[8:16])
        directory = json.loads(handle.read(json_bytes))

        old_id = directory["identity"]["weights_id"]
        if old_id == new_id:
            print(f"already tagged {new_id!r}; nothing to do")
            return 0

        directory["identity"]["weights_id"] = new_id
        # Compact, exactly as the converter writes it: any extra whitespace here
        # is padding that eats the slack this edit depends on.
        encoded = json.dumps(directory, separators=(",", ":")).encode()

        def payload_start(size: int) -> int:
            end = PREFIX_BYTES + size
            return -(-end // PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT

        before, after = payload_start(json_bytes), payload_start(len(encoded))
        if before != after:
            print(
                f"{path}: refusing to retag - payload would move "
                f"{before:,} -> {after:,}, which needs a full rewrite",
                file=sys.stderr,
            )
            return 1

        handle.seek(8)
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(b"\0" * (before - PREFIX_BYTES - len(encoded)))

    print(f"weights_id {old_id!r} -> {new_id!r}  (payload unchanged at {before:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
