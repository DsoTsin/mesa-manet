import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("timing", Path(__file__).parents[1] / "panvk_pass_timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def pair(kind, start, end, subqueue, label="SNL/gbuffer", number=1):
    return [
        {"event": f"begin_{kind}", "time_ns": str(start), "params": {}},
        {"event": f"end_{kind}", "time_ns": str(end), "params": {
            "subqueue": str(subqueue), "command_buffer": "0x123", "pass": str(number), "label": label}},
    ]


def frame(events, number=42):
    return {"frame": number, "batches": [{"events": events}]}


class PassTimingTests(unittest.TestCase):
    def test_parallel_phases_are_not_added(self):
        data = frame(pair("render", 100, 300, 0))
        data["batches"].append({"events": pair("render", 200, 500, 1)})
        report = timing.summarize(timing.extract_passes([data]))
        self.assertEqual(report["frames"][0]["interval_union_ns"], 400)
        self.assertEqual([p["duration_ns"] for p in report["passes"]], [200, 300])
        self.assertEqual(len(report["summary"]), 2)

    def test_nested_passes_and_escaped_labels(self):
        outer = pair("render", 100, 500, 0, 'SNL/"pass"\\x\n')
        inner = pair("render", 200, 300, 0, "resolve", 2)
        passes = timing.extract_passes([frame([outer[0], *inner, outer[1]])])
        self.assertEqual([p["pass"] for p in passes], [2, 1])
        self.assertIn('SNL/\\"pass\\"\\\\x\\n', timing.text_report(timing.summarize(passes)))

    def test_frame_filter_and_repeated_command_buffer(self):
        data = [frame(pair("dispatch", 100, 200, 2), 41),
                frame(pair("dispatch", 300, 400, 2), 42)]
        self.assertEqual(len(timing.extract_passes(data, 42, 42)), 1)
        self.assertEqual(timing.summarize(timing.extract_passes(data))["summary"][0]["count"], 2)

    def test_indirect_dispatch(self):
        p = timing.extract_passes([frame(pair("dispatch_indirect", 100, 900, 2))])[0]
        self.assertEqual(p["duration_ns"], 800)

    def test_invalid_traces_are_rejected(self):
        cases = [pair("render", 0, 100, 0), pair("render", 100, 100, 0),
                 pair("render", 200, 100, 0), pair("render", 100, 200, 9),
                 pair("render", 100, 200, 0)[:1], pair("render", 100, 200, 0)[1:],
                 [pair("render", 100, 200, 0)[0], pair("dispatch", 100, 200, 2)[1]]]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ValueError):
                timing.extract_passes([frame(events)])

    def test_union_handles_gaps_and_nested_intervals(self):
        self.assertEqual(timing.interval_union([(10, 100), (20, 30), (50, 70), (150, 200)]), 140)


if __name__ == "__main__":
    unittest.main()
