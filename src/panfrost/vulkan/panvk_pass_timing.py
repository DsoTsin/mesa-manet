import argparse
import collections
import json
from pathlib import Path


KINDS = {"render", "dispatch", "dispatch_indirect"}
SUBQUEUES = {0: "vertex_tiler", 1: "fragment", 2: "compute"}


def interval_union(intervals):
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def extract_passes(frames, first_frame=None, last_frame=None):
    passes = []
    for frame in frames:
        frame_id = int(frame["frame"])
        if first_frame is not None and frame_id < first_frame:
            continue
        if last_frame is not None and frame_id > last_frame:
            continue
        for batch_id, batch in enumerate(frame["batches"]):
            stack = []
            for event in batch["events"]:
                direction, _, kind = event["event"].partition("_")
                if kind not in KINDS:
                    continue
                timestamp = int(event["time_ns"])
                if direction == "begin":
                    stack.append((kind, timestamp))
                    continue
                if direction != "end":
                    continue
                if not stack or stack[-1][0] != kind:
                    raise ValueError(f"unmatched {event['event']} in frame {frame_id}, batch {batch_id}")
                _, start = stack.pop()
                if start == 0 or timestamp <= start:
                    raise ValueError(f"invalid GPU timestamps in frame {frame_id}, batch {batch_id}")
                params = event["params"]
                subqueue = int(params["subqueue"])
                if subqueue not in SUBQUEUES:
                    raise ValueError(f"unknown subqueue {subqueue}")
                passes.append({
                    "frame": frame_id,
                    "batch": batch_id,
                    "command_buffer": params["command_buffer"],
                    "pass": int(params["pass"]),
                    "kind": kind,
                    "subqueue": SUBQUEUES[subqueue],
                    "label": params.get("label", ""),
                    "begin_ns": start,
                    "end_ns": timestamp,
                    "duration_ns": timestamp - start,
                    "params": params,
                })
            if stack:
                raise ValueError(f"incomplete pass in frame {frame_id}, batch {batch_id}")
    return passes


def summarize(passes):
    groups = collections.defaultdict(list)
    frames = collections.defaultdict(list)
    for item in passes:
        groups[(item["label"], item["kind"], item["subqueue"])].append(item["duration_ns"])
        frames[item["frame"]].append((item["begin_ns"], item["end_ns"]))
    summaries = []
    for (label, kind, subqueue), durations in sorted(groups.items()):
        summaries.append({
            "label": label,
            "kind": kind,
            "subqueue": subqueue,
            "count": len(durations),
            "mean_ns": sum(durations) / len(durations),
            "min_ns": min(durations),
            "max_ns": max(durations),
        })
    return {
        "schema": "panvk-pass-timing-v1",
        "measurement": "GPU queue intervals including dependencies; overlapping intervals are not additive",
        "passes": passes,
        "summary": summaries,
        "frames": [{
            "frame": frame,
            "span_ns": max(stop for _, stop in intervals) - min(start for start, _ in intervals),
            "interval_union_ns": interval_union(intervals),
        } for frame, intervals in sorted(frames.items())],
    }


def text_report(report):
    lines = ["GPU pass intervals (ms, includes GPU waits; phases may overlap)",
             "Frame Batch Pass Subqueue      Kind                 Duration Label"]
    for p in report["passes"]:
        label = json.dumps(p["label"], ensure_ascii=False)
        lines.append(f"{p['frame']:5} {p['batch']:5} {p['pass']:4} {p['subqueue']:13} "
                     f"{p['kind']:20} {p['duration_ns'] / 1e6:8.4f} {label}")
    lines.append("\nCount     Mean      Min      Max Subqueue      Kind                 Label")
    for s in report["summary"]:
        label = json.dumps(s["label"], ensure_ascii=False)
        lines.append(f"{s['count']:5} {s['mean_ns'] / 1e6:8.4f} {s['min_ns'] / 1e6:8.4f} "
                     f"{s['max_ns'] / 1e6:8.4f} {s['subqueue']:13} {s['kind']:20} {label}")
    for f in report["frames"]:
        lines.append(f"Frame {f['frame']}: span {f['span_ns'] / 1e6:.4f} ms, "
                     f"interval union {f['interval_union_ns'] / 1e6:.4f} ms")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize PanVK MESA_GPU_TRACES=print_json GPU pass intervals")
    parser.add_argument("input", type=Path, help="MESA_GPU_TRACEFILE from a normally closed Vulkan device")
    parser.add_argument("--json", type=Path, help="write per-pass records and grouped statistics")
    parser.add_argument("--text", type=Path, help="write the text report (default: stdout)")
    parser.add_argument("--first-frame", type=int)
    parser.add_argument("--last-frame", type=int)
    args = parser.parse_args()
    try:
        frames = json.loads(args.input.read_text(encoding="utf-8"))
        passes = extract_passes(frames, args.first_frame, args.last_frame)
        if not passes:
            raise ValueError("no GPU pass records in the selected frames")
        report = summarize(passes)
        if args.json:
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        text = text_report(report)
        if args.text:
            args.text.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"invalid PanVK trace: {error}\n")


if __name__ == "__main__":
    main()
