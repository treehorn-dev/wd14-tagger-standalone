from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tagger import __version__

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command")

    tag_parser = subparsers.add_parser("tag", add_help=False)
    group = tag_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", dest="dir", help="Predictions for all images in the directory")
    group.add_argument("--file", dest="file", help="Predictions for one file")
    tag_parser.add_argument("--threshold", type=float, default=0.35)
    tag_parser.add_argument("--ext", default=".txt")
    tag_parser.add_argument("--overwrite", action="store_true")
    tag_parser.add_argument("--cpu", action="store_true")
    tag_parser.add_argument("--rawtag", action="store_true")
    tag_parser.add_argument("--recursive", action="store_true")
    tag_parser.add_argument("--exclude-tag", dest="exclude_tags", action="append")
    tag_parser.add_argument("--additional-tag", dest="additional_tags", action="append")
    tag_parser.add_argument("--model", default="wd14-convnextv2.v1")
    return parser


def build_root_payload(raw: str) -> dict[str, Any]:
    return success_response(
        raw=raw,
        parsed={"path": [], "options": {}, "flags": {}},
        result={
            "description": "WD14 tagger CLI",
            "commands": [
                {
                    "name": "tag",
                    "description": "Tag a file or directory with the selected WD14 model.",
                    "usage": "wd14-tagger tag --file <image>|--dir <folder> [options]",
                }
            ],
        },
        next_actions=[
            {
                "command": "wd14-tagger tag --file /path/to/image.jpg",
                "description": "Tag a single image file.",
            },
            {
                "command": "wd14-tagger tag --dir /path/to/frames --recursive",
                "description": "Tag all images in a directory tree.",
            },
        ],
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    raw = command_raw(argv)
    if not argv:
        emit(build_root_payload(raw))
        return 0

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        emit(
            error_response(
                raw=raw,
                parsed={"path": argv[:1], "options": {}, "flags": {}},
                code="INVALID_ARGUMENTS",
                message="Invalid command arguments.",
                fix="Run the root command to inspect the available command tree and tag usage.",
                next_actions=[
                    {"command": "wd14-tagger", "description": "Show the root command tree."}
                ],
            )
        )
        return 2

    if args.command == "tag":
        parsed = parsed_tag_command(args)
        try:
            result = execute_tag(parsed)
        except FileNotFoundError as exc:
            emit(
                error_response(
                    raw=raw,
                    parsed=parsed,
                    code="FILE_NOT_FOUND",
                    message=str(exc),
                    fix="Verify the input path exists and is readable, then run the tag command again.",
                    next_actions=[
                        {"command": "wd14-tagger", "description": "Show the root command tree."}
                    ],
                )
            )
            return 1
        except Exception as exc:
            emit(
                error_response(
                    raw=raw,
                    parsed=parsed,
                    code="TAGGING_FAILED",
                    message=str(exc),
                    fix="Check the selected model assets and runtime dependencies, then retry the tag command.",
                    next_actions=[
                        {"command": "wd14-tagger", "description": "Show the root command tree."}
                    ],
                )
            )
            return 1

        emit(
            success_response(
                raw=raw,
                parsed=parsed,
                result=result,
                next_actions=next_actions_for_result(parsed, result),
            )
        )
        return 0

    emit(build_root_payload(raw))
    return 0


def command_raw(argv: list[str]) -> str:
    return " ".join(["wd14-tagger", *argv]).strip()


def parsed_tag_command(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "path": ["tag"],
        "options": {
            "file": args.file,
            "dir": args.dir,
            "threshold": args.threshold,
            "ext": args.ext,
            "exclude_tags": args.exclude_tags or [],
            "additional_tags": args.additional_tags or [],
            "model": args.model,
        },
        "flags": {
            "overwrite": bool(args.overwrite),
            "cpu": bool(args.cpu),
            "rawtag": bool(args.rawtag),
            "recursive": bool(args.recursive),
        },
    }


def success_response(raw: str, parsed: dict[str, Any], result: dict[str, Any], next_actions: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "ok": True,
        "command": {
            "raw": raw,
            "parsed": parsed,
            "resolved": {
                "executable": "wd14-tagger",
                "cwd": os.getcwd(),
                "version": __version__,
            },
        },
        "result": result,
        "next_actions": next_actions,
    }


def error_response(raw: str, parsed: dict[str, Any], code: str, message: str, fix: str, next_actions: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "ok": False,
        "command": {
            "raw": raw,
            "parsed": parsed,
            "resolved": {
                "executable": "wd14-tagger",
                "cwd": os.getcwd(),
                "version": __version__,
            },
        },
        "error": {"code": code, "message": message},
        "fix": fix,
        "next_actions": next_actions,
    }


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def next_actions_for_result(parsed: dict[str, Any], result: dict[str, Any]) -> list[dict[str, str]]:
    if result.get("mode") == "file":
        path = parsed["options"]["file"]
        return [
            {"command": f"wd14-tagger tag --file {path}", "description": "Re-run tagging for this image."},
            {"command": "wd14-tagger", "description": "Show the root command tree."},
        ]
    target = parsed["options"]["dir"]
    return [
        {"command": f"wd14-tagger tag --dir {target} --recursive", "description": "Re-run recursive directory tagging."},
        {"command": "wd14-tagger", "description": "Show the root command tree."},
    ]


def execute_tag(parsed: dict[str, Any]) -> dict[str, Any]:
    model_name = parsed["options"]["model"]
    interrogator = get_interrogator(model_name, parsed["flags"]["cpu"])
    exclude_tags = parse_tag_list(parsed["options"]["exclude_tags"], reverse_escape=True)
    additional_tags = parse_tag_list(parsed["options"]["additional_tags"], reverse_escape=False)

    if parsed["options"]["file"]:
        image_path = Path(parsed["options"]["file"])
        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))
        tags = interrogate_image(
            image_path=image_path,
            interrogator=interrogator,
            threshold=parsed["options"]["threshold"],
            rawtag=parsed["flags"]["rawtag"],
            exclude_tags=exclude_tags,
            additional_tags=additional_tags,
        )
        return {
            "mode": "file",
            "input": str(image_path),
            "model": model_name,
            "tags": list(tags.keys()),
            "tag_scores": tags,
        }

    root_path = Path(parsed["options"]["dir"])
    if not root_path.is_dir():
        raise FileNotFoundError(str(root_path))

    processed = 0
    skipped = 0
    outputs: list[dict[str, Any]] = []
    for image_path in iter_image_files(root_path, recursive=parsed["flags"]["recursive"]):
        caption_path = image_path.parent / f"{image_path.stem}{parsed['options']['ext']}"
        if caption_path.is_file() and not parsed["flags"]["overwrite"]:
            skipped += 1
            continue
        tags = interrogate_image(
            image_path=image_path,
            interrogator=interrogator,
            threshold=parsed["options"]["threshold"],
            rawtag=parsed["flags"]["rawtag"],
            exclude_tags=exclude_tags,
            additional_tags=additional_tags,
        )
        caption_path.write_text(", ".join(tags.keys()))
        processed += 1
        if len(outputs) < 10:
            outputs.append({"image": str(image_path), "caption": str(caption_path), "tags": list(tags.keys())})

    return {
        "mode": "dir",
        "input": str(root_path),
        "model": model_name,
        "processed": processed,
        "skipped": skipped,
        "outputs": outputs,
        "truncated": processed > len(outputs),
    }


def parse_tag_list(values: list[str], reverse_escape: bool) -> list[str] | set[str]:
    tags: list[str] = []
    for value in values:
        for tag in value.split(","):
            cleaned = tag.strip()
            if cleaned:
                tags.append(cleaned)
    if reverse_escape:
        reverse_escaped = [tag.replace(" ", "_").replace(r"\(", "(").replace(r"\)", ")") for tag in tags]
        return set([*tags, *reverse_escaped])
    return list(dict.fromkeys(tags))


def iter_image_files(folder_path: Path, recursive: bool):
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        elif recursive and path.is_dir():
            yield from iter_image_files(path, recursive=True)


def get_interrogator(model_name: str, cpu: bool):
    from tagger.interrogators import interrogators

    interrogator = interrogators[model_name]
    if cpu:
        interrogator.use_cpu()
    return interrogator


def interrogate_image(image_path: Path, interrogator, threshold: float, rawtag: bool, exclude_tags, additional_tags):
    from PIL import Image, ImageFile
    from tagger.interrogator.interrogator import AbsInterrogator

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(image_path) as image:
        result = interrogator.interrogate(image)

    return AbsInterrogator.postprocess_tags(
        result[1],
        threshold=threshold,
        escape_tag=not rawtag,
        replace_underscore=not rawtag,
        exclude_tags=exclude_tags,
        additional_tags=list(additional_tags),
    )


def main_entry() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    main_entry()
