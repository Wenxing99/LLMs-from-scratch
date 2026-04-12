import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streamingly replace case-insensitive 'MiniMind' with 'FeiFeiMind' in a JSONL file."
    )
    parser.add_argument("src", type=Path, help="Source JSONL file")
    parser.add_argument("dst", type=Path, help="Destination JSONL file")
    parser.add_argument(
        "--old", default="MiniMind", help="Case-insensitive source token to replace"
    )
    parser.add_argument(
        "--new", default="FeiFeiMind", help="Replacement token to write"
    )
    parser.add_argument(
        "--progress-every",
        default=500000,
        type=int,
        help="Print one progress line every N processed lines",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Source file not found: {args.src}")
    if args.dst.exists():
        raise FileExistsError(f"Destination file already exists: {args.dst}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(re.escape(args.old), flags=re.IGNORECASE)
    line_count = 0
    touched_lines = 0
    replace_count = 0

    with args.src.open("r", encoding="utf-8", newline="") as src, args.dst.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line_count, line in enumerate(src, start=1):
            match_count = len(pattern.findall(line))
            if match_count:
                touched_lines += 1
                replace_count += match_count
                line = pattern.sub(args.new, line)
            dst.write(line)

            if args.progress_every > 0 and line_count % args.progress_every == 0:
                print(
                    f"progress lines={line_count} touched_lines={touched_lines} replacements={replace_count}"
                )

    print(
        f"done lines={line_count} touched_lines={touched_lines} replacements={replace_count}"
    )


if __name__ == "__main__":
    main()
