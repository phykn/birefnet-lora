import argparse
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiRefNet-LoRA inference API")
    parser.add_argument("--host", required=True, help="bind host, e.g. 0.0.0.0")
    parser.add_argument("--port", required=True, type=int, help="bind port")
    parser.add_argument(
        "--weight",
        required=True,
        help="path to trained LoRA adapter checkpoint (.pth)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["BIREFNET_LORA_WEIGHT"] = args.weight
    uvicorn.run("src.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
