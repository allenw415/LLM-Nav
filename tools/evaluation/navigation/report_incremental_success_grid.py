from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


RATIO_BINS = ("1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0")
ROOM_COUNTS = (2, 3, 4, 5, 6)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(dataset: dict[str, dict], records: list[dict]) -> dict:
    cells: dict[tuple[int, str], list[dict]] = {
        (room_count, ratio): []
        for room_count in ROOM_COUNTS
        for ratio in RATIO_BINS
    }
    errors = 0
    for record in records:
        if record.get("error") is not None:
            errors += 1
            continue
        test = dataset[record["test_id"]]
        cells[(int(test["visited_room_count"]), str(test["ratio_stratum"]))].append(record)

    def cell_value(items: list[dict]) -> dict:
        success_count = sum(record.get("success") is True for record in items)
        return {
            "success_count": success_count,
            "valid_count": len(items),
            "rate": success_count / len(items) if items else None,
        }

    grid = {
        room_count: {
            ratio: cell_value(cells[(room_count, ratio)])
            for ratio in RATIO_BINS
        }
        for room_count in ROOM_COUNTS
    }
    row_overall = {
        room_count: cell_value(
            [record for ratio in RATIO_BINS for record in cells[(room_count, ratio)]]
        )
        for room_count in ROOM_COUNTS
    }
    column_overall = {
        ratio: cell_value(
            [record for room_count in ROOM_COUNTS for record in cells[(room_count, ratio)]]
        )
        for ratio in RATIO_BINS
    }
    overall = cell_value([record for items in cells.values() for record in items])
    return {
        "grid": grid,
        "row_overall": row_overall,
        "column_overall": column_overall,
        "overall": overall,
        "raw_count": len(records),
        "error_count": errors,
    }


def format_rate(value: dict) -> str:
    if value["rate"] is None:
        return "—"
    return f"{100.0 * value['rate']:.1f}%"


def format_cell(value: dict) -> str:
    if value["rate"] is None:
        return "—"
    return format_rate(value)


def html_method_row(label: str, method: str, summary: dict, room_count: int | None) -> str:
    values = [summary["grid"][room_count][ratio] for ratio in RATIO_BINS] if room_count else [
        summary["column_overall"][ratio] for ratio in RATIO_BINS
    ]
    values.append(summary["row_overall"][room_count] if room_count else summary["overall"])
    cells = "".join(f'<td class="{method}">{format_cell(value)}</td>' for value in values)
    room_cell = f'<td class="room" rowspan="2">{room_count}</td>' if room_count else '<td class="room overall-room" rowspan="2">Overall</td>'
    return f"<tr>{room_cell}<td class=\"method {method}\">{html.escape(label)}</td>{cells}</tr>"


def build_html(
    *,
    title: str,
    without: dict,
    with_trajectory: dict,
    output_csv: str,
) -> str:
    rows = []
    for room_count in ROOM_COUNTS:
        rows.append(html_method_row("w/o traj.", "without", without, room_count))
        rows.append(html_method_row("with traj.", "with", with_trajectory, room_count))
    rows.append(html_method_row("w/o traj.", "without", without, None))
    rows.append(html_method_row("with traj.", "with", with_trajectory, None))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ margin: 30px; font-family: Arial, sans-serif; color: #111; }}
h1 {{ font-size: 34px; font-weight: 400; margin: 0 0 34px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1700px; table-layout: fixed; font-size: 25px; }}
th {{ background: #1f4e79; color: white; padding: 15px 8px; text-align: center; }}
th:first-child {{ width: 230px; text-align: left; }}
th:nth-child(2) {{ width: 180px; }}
td {{ padding: 9px 8px; text-align: center; border-bottom: 1px solid #b9c6d2; }}
td.room {{ background: #f5f7f9; font-weight: 700; text-align: left; vertical-align: middle; }}
td.method {{ color: white; font-weight: 700; text-align: left; }}
td.without {{ background: #687282; }}
td.with {{ background: #2f75b5; }}
td:not(.method):not(.room).without {{ background: #eeeeee; color: #111; }}
td:not(.method):not(.room).with {{ background: #d9eaf7; color: #111; }}
tr:nth-last-child(-n+2) td:not(.method):not(.room) {{ font-weight: 700; }}
tr:nth-last-child(-n+2) td:not(.method):not(.room):last-child {{ color: #d00000; }}
small {{ display: block; color: #555; font-size: 12px; font-weight: 400; margin-top: 2px; }}
.note {{ max-width: 1700px; margin-top: 18px; color: #444; font-size: 15px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<table><thead><tr>
<th>Visited Room Count ↓<br>/ Detour Ratio →</th><th>Method</th>
{''.join(f'<th>{ratio}</th>' for ratio in RATIO_BINS)}<th>Overall</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="note">Each percentage includes the numerator/valid denominator in small text. API-error cases are excluded from the success-rate denominator and reported separately in the CSV. <a href="{html.escape(output_csv)}">Download CSV</a>.</p>
</body></html>
"""


def write_csv(path: Path, without: dict, with_trajectory: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["visited_room_count", "method", *RATIO_BINS, "overall", "raw_count", "api_error_count"])
        for method, summary in (("w/o traj.", without), ("with traj.", with_trajectory)):
            for room_count in ROOM_COUNTS:
                writer.writerow([
                    room_count,
                    method,
                    *[
                        f"{format_rate(summary['grid'][room_count][ratio])} ({summary['grid'][room_count][ratio]['success_count']}/{summary['grid'][room_count][ratio]['valid_count']})"
                        if summary['grid'][room_count][ratio]['rate'] is not None else "—"
                        for ratio in RATIO_BINS
                    ],
                    f"{format_rate(summary['row_overall'][room_count])} ({summary['row_overall'][room_count]['success_count']}/{summary['row_overall'][room_count]['valid_count']})",
                    summary["raw_count"],
                    summary["error_count"],
                ])
            overall = summary["overall"]
            writer.writerow([
                "Overall",
                method,
                *[
                    f"{format_rate(summary['column_overall'][ratio])} ({summary['column_overall'][ratio]['success_count']}/{summary['column_overall'][ratio]['valid_count']})"
                    if summary['column_overall'][ratio]['rate'] is not None else "—"
                    for ratio in RATIO_BINS
                ],
                f"{format_rate(overall)} ({overall['success_count']}/{overall['valid_count']})",
                summary["raw_count"],
                summary["error_count"],
            ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--without-results", required=True)
    parser.add_argument("--with-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Navigation Success Rate by Visited Room Count and Detour Ratio")
    args = parser.parse_args()
    dataset = {record["test_id"]: record for record in load_jsonl(Path(args.dataset))}
    without = summarize(dataset, load_jsonl(Path(args.without_results)))
    with_trajectory = summarize(dataset, load_jsonl(Path(args.with_results)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "success_grid.csv"
    html_path = output_dir / "success_grid.html"
    write_csv(csv_path, without, with_trajectory)
    html_path.write_text(
        build_html(
            title=args.title,
            without=without,
            with_trajectory=with_trajectory,
            output_csv=csv_path.name,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"without_trajectory": without, "with_trajectory": with_trajectory}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
