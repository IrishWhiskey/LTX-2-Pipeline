#!/usr/bin/env python3
"""CLI to run predictions on the irishwhiskey/test Replicate model."""

import argparse
import os
import sys

import replicate


MODEL = "irishwhiskey/test"
DEFAULT_PROMPT_FILE = "test_prompt.txt"


def load_env(path=".env.local"):
    """Load KEY=VALUE pairs from a dotenv file."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


def main():
    parser = argparse.ArgumentParser(description=f"Run a prediction on {MODEL}")
    parser.add_argument("prompt", nargs="?", help="Text prompt (reads test_prompt.txt if omitted)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Poll until prediction completes")
    args = parser.parse_args()

    load_env()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN not set. Add it to .env.local or export it.", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt
    if not prompt:
        if os.path.isfile(DEFAULT_PROMPT_FILE):
            prompt = open(DEFAULT_PROMPT_FILE).read().strip()
        else:
            print(f"Error: No prompt given and {DEFAULT_PROMPT_FILE} not found.", file=sys.stderr)
            sys.exit(1)

    print(f"Running {MODEL} with prompt: {prompt!r}")

    input_params = {
        "prompt": prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "seed": args.seed,
        "enhance_prompt": args.enhance_prompt,
    }

    model = replicate.models.get(MODEL)
    version = model.latest_version

    if args.wait:
        output = replicate.run(f"{MODEL}:{version.id}", input=input_params)
        print(f"Output: {output}")
    else:
        prediction = replicate.predictions.create(version=version.id, input=input_params)
        print(f"Prediction created: {prediction.id}")
        print(f"Status: {prediction.status}")
        print(f"URL: https://replicate.com/p/{prediction.id}")


if __name__ == "__main__":
    main()
