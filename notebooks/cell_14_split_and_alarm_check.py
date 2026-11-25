# ===== 14. FL3D split check + alarm evaluation on the test videos =====
# Needs: setup cells, part 5 (FL3D) and the video logs from part 12 (already on Drive).

# ---------- A. FL3D split by video: table + overlap check ----------
assert len(fl3d) and len(fl3d_groups), "Run part 5 (FL3D) first."
vid = np.array(fl3d_groups)
y = np.array([f[0] for f in fl3d])
split_of = {}
for name, ids in [("train", fl3d_fit), ("validation", fl3d_val), ("test", fl3d_test)]:
    for v in set(vid[ids]):
        split_of.setdefault(v, set()).add(name)
rows = []
for v in sorted(split_of):
    m = vid == v
    rows.append({"video": v, "split": ", ".join(sorted(split_of[v])),
                 "closed_frames": int((y[m] == 0).sum()), "open_frames": int((y[m] == 1).sum())})
split_table = pd.DataFrame(rows).sort_values(["split", "video"]).reset_index(drop=True)
split_table.to_csv(f"{RESULTS}/fl3d_split_videos.csv", index=False)

tr_v, va_v, te_v = set(vid[fl3d_fit]), set(vid[fl3d_val]), set(vid[fl3d_test])
print("Videos per split -> train:", len(tr_v), "| validation:", len(va_v), "| test:", len(te_v))
print("Overlap train/test:", len(tr_v & te_v), "| train/validation:", len(tr_v & va_v), "| validation/test:", len(va_v & te_v))
assert not (tr_v & te_v) and not (tr_v & va_v) and not (va_v & te_v), "A video appears in two splits!"
print("OK: no video has frames in more than one split.")
print("Note: the split is by VIDEO. NITYMED file names do not give the driver ID, so the same driver")
print("      in two splits cannot be ruled out from the file names alone.")
display(split_table.groupby("split")[["closed_frames", "open_frames"]].sum().assign(videos=split_table.groupby("split").size()))
display(split_table)

# ---------- B. Ground-truth microsleep events from the FL3D frame labels ----------
def frame_number(p):
    nums = re.findall(r"\d+", Path(p).stem)
    return int(nums[-1]) if nums else None

frames_by_video = defaultdict(list)          # all labelled frames (alert / microsleep / yawning)
for p, lab in zip(fl3d_paths, fl3d_labels):
    n = frame_number(p)
    if n is not None:
        frames_by_video[Path(p).parent.name].append((n, lab))

video_files = {p.stem: p for p in Path(NITYMED_DIR).rglob("*.mp4")} if NITYMED_DIR else {}
event_files = sorted(Path(LOGS).glob("video_*_events.json"))
test_logs = [f for f in event_files if f.stem[len("video_"):-len("_events")] in te_v]
print("\nTest videos with a system log:", [f.stem[6:-7] for f in test_logs])
assert test_logs, "No video logs for FL3D test videos: run part 12 first."

GAP_S = 0.5           # labelled microsleep frames closer than this belong to one event
MATCH_TOL_S = 0.5     # a detection may start/end this close to the labelled event
DETECT_S = 0.5        # closure length the system needs before it is more than a blink
ALARM_S = 1.5         # closure length that raises the alarm

per_event, per_video = [], []
for f in test_logs:
    stem = f.stem[len("video_"):-len("_events")]
    frs = sorted(frames_by_video.get(stem, []))
    if not frs or stem not in video_files:
        print("skip", stem, "(no labels or no video file)"); continue
    cap = cv2.VideoCapture(str(video_files[stem]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    n_max = max(n for n, _ in frs)
    step = n_video / (n_max + 1) if n_max > 0 else 1.0
    step = round(step) if abs(step - round(step)) < 0.1 * max(1, round(step)) else step
    print(f"{stem}: {len(frs)} labelled frames, frame numbers 0..{n_max}, video has {n_video} frames "
          f"at {fps:.1f} fps -> 1 label step = {step} video frames. Example files: "
          f"{[Path(p).name for p in fl3d_paths if Path(p).parent.name == stem][:3]}")
    times = [(n * step / fps, lab) for n, lab in frs]
    spacing = np.median(np.diff([t for t, _ in times])) if len(times) > 1 else 1 / fps
    gap = max(GAP_S, 3 * spacing)
    span_s = times[-1][0] - times[0][0] + spacing

    # labelled microsleep events
    gt, cur = [], None
    for t, lab in times:
        if lab == 0:
            if cur and t - cur[1] <= gap:
                cur[1] = t
            else:
                cur = [t, t]; gt.append(cur)
        elif lab == 1 and cur and t - cur[1] > 0:
            cur = None
    gt = [(s, e + spacing) for s, e in gt]

    # system closures from the log (microsleep = 0.5-1.5 s, long_closure >= 1.5 s)
    ev = json.load(open(f))["events"]
    sys_cl = [(e["time_s"] - e["duration_s"], e["time_s"], e["event"]) for e in ev
              if e["event"] in ("microsleep", "long_closure")]
    sys_cl = [c for c in sys_cl if times[0][0] - MATCH_TOL_S <= c[0] <= times[-1][0] + MATCH_TOL_S]

    def overlaps(a0, a1, b0, b1):
        return a0 <= b1 + MATCH_TOL_S and b0 <= a1 + MATCH_TOL_S

    used_det, used_alarm = set(), set()
    for gs, ge in gt:
        det = [i for i, (s, e, k) in enumerate(sys_cl) if overlaps(gs, ge, s, e)]
        alarm = [i for i in det if sys_cl[i][2] == "long_closure"]
        used_det.update(det); used_alarm.update(alarm)
        first = min(det, key=lambda i: sys_cl[i][0]) if det else None
        first_alarm = min(alarm, key=lambda i: sys_cl[i][0]) if alarm else None
        per_event.append({
            "video": stem, "event_start_s": round(gs, 2), "event_length_s": round(ge - gs, 2),
            "detected": first is not None,
            "detect_delay_s": round(sys_cl[first][0] + DETECT_S - gs, 2) if first is not None else None,
            "alarm": first_alarm is not None,
            "alarm_delay_s": round(sys_cl[first_alarm][0] + ALARM_S - gs, 2) if first_alarm is not None else None})
    false_det = [c for i, c in enumerate(sys_cl) if i not in used_det]
    false_alarm = [c for i, c in enumerate(sys_cl) if c[2] == "long_closure" and i not in used_alarm and i not in used_det]
    long_gt = [ge - gs for gs, ge in gt if ge - gs >= ALARM_S]
    per_video.append({"video": stem, "labelled_minutes": round(span_s / 60, 2), "true_events": len(gt),
                      "true_events_>=1.5s": len(long_gt),
                      "detected": sum(e["detected"] for e in per_event if e["video"] == stem),
                      "false_detections": len(false_det), "false_alarms": len(false_alarm)})

per_event = pd.DataFrame(per_event); per_video = pd.DataFrame(per_video)
per_event.to_csv(f"{RESULTS}/alarm_events.csv", index=False)
per_video.to_csv(f"{RESULTS}/alarm_per_video.csv", index=False)

hours = per_video.labelled_minutes.sum() / 60
long_ev = per_event[per_event.event_length_s >= ALARM_S]
summary = {
    "test_videos": len(per_video),
    "labelled_minutes": round(per_video.labelled_minutes.sum(), 1),
    "true_microsleep_events": int(len(per_event)),
    "event_recall_(closure>=0.5s)": round(per_event.detected.mean(), 3) if len(per_event) else None,
    "median_detect_delay_s": round(per_event.detect_delay_s.median(), 2) if per_event.detected.any() else None,
    "false_detections_per_hour": round(per_video.false_detections.sum() / hours, 1) if hours else None,
    "true_events_>=1.5s": int(len(long_ev)),
    "alarm_recall_(events>=1.5s)": round(long_ev.alarm.mean(), 3) if len(long_ev) else None,
    "median_alarm_delay_s": round(long_ev.alarm_delay_s.median(), 2) if long_ev.alarm.any() else None,
    "false_alarms_per_hour": round(per_video.false_alarms.sum() / hours, 1) if hours else None,
}
json.dump(summary, open(f"{RESULTS}/alarm_evaluation.json", "w"), indent=2)
display(per_video)
display(per_event)
print(json.dumps(summary, indent=2))
print(f"\nOnly {per_video.labelled_minutes.sum():.1f} labelled minutes: per-hour rates are rough estimates.")
print("Saved: results/fl3d_split_videos.csv, alarm_events.csv, alarm_per_video.csv, alarm_evaluation.json")
