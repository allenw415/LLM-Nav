from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from control import StreetViewController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Street View web app through StreetViewController."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium in headless mode.",
    )
    parser.add_argument(
        "--screenshot",
        default="sv.png",
        help="Write a pano screenshot to this path. Default: sv.png",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=5.0,
        help="Keep the browser open for this many seconds after actions. Default: 5",
    )
    parser.add_argument(
        "--action",
        action="append",
        choices=["front", "left", "right", "up", "level", "down", "zin", "zout"],
        help="Optional action to perform. Can be passed multiple times.",
    )
    return parser


def run_action(controller: StreetViewController, action: str) -> None:
    if action == "front":
        moved = controller.action_front()
        print(f"[action] front -> moved={moved}")
        return
    if action == "left":
        controller.action_turn_left()
        print("[action] left")
        return
    if action == "right":
        controller.action_turn_right()
        print("[action] right")
        return
    if action == "up":
        controller.action_pitch_up()
        print("[action] up")
        return
    if action == "level":
        controller.action_pitch_level()
        print("[action] level")
        return
    if action == "down":
        controller.action_pitch_down()
        print("[action] down")
        return
    if action == "zin":
        controller.action_zoom_in()
        print("[action] zin")
        return
    if action == "zout":
        controller.action_zoom_out()
        print("[action] zout")
        return
    raise ValueError(f"Unsupported action: {action}")


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[3]
    web_root = Path(__file__).resolve().parent / "web"
    screenshot_path = (project_root / args.screenshot).resolve()

    print(f"[info] web_root={web_root}")
    print(f"[info] headless={args.headless}")

    with StreetViewController(web_root=web_root, headless=args.headless) as controller:
        print("[status]", controller.get_status_text())

        for action in args.action or []:
            run_action(controller, action)
            print("[status]", controller.get_status_text())

        screenshot_path.write_bytes(controller.screenshot_pano())
        print(f"[saved] screenshot -> {screenshot_path}")

        errors = controller.get_errors(clear=False)
        if errors:
            print("[errors]")
            for entry in errors:
                print(f"  - {entry.kind}: {entry.text}")

        if args.wait_seconds > 0:
            print(f"[info] waiting {args.wait_seconds:.1f}s before exit")
            time.sleep(args.wait_seconds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
