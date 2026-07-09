#!/usr/bin/env python3
"""Convert the selected model's source weights into Kipp's supported GGUF.

This tooling-only entry point will become model-specific after the first
target family and exact tensor layout are selected.
"""


def main() -> None:
    raise SystemExit("No Kipp target model has been selected; conversion is unavailable.")


if __name__ == "__main__":
    main()
