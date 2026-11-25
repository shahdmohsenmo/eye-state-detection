# ===== 14b. A closer look: short labelled events and the false alarms =====
# Run right after cell 14 (same session).
pe = pd.read_csv(f"{RESULTS}/alarm_events.csv")
long_enough = pe[pe.event_length_s >= DETECT_S]
short = pe[pe.event_length_s < DETECT_S]
print(f"Labelled microsleeps >= 0.5 s: {len(long_enough)}, detected {int(long_enough.detected.sum())} "
      f"({long_enough.detected.mean():.1%}), median delay {long_enough.detect_delay_s.median():.2f} s")
print(f"Labelled microsleeps <  0.5 s (blink length): {len(short)}, detected {int(short.detected.sum())}")

NAMES = {1: "alert", 0: "microsleep", -1: "yawning"}
unmatched, pics = [], []
for f in test_logs:
    stem = f.stem[len("video_"):-len("_events")]
    lab_at = dict(frames_by_video.get(stem, []))
    gts = pe[pe.video == stem]
    cap = cv2.VideoCapture(str(video_files[stem])); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for e in json.load(open(f))["events"]:
        if e["event"] not in ("microsleep", "long_closure"):
            continue
        s, t_end = e["time_s"] - e["duration_s"], e["time_s"]
        hit = ((gts.event_start_s <= t_end + MATCH_TOL_S) &
               (s <= gts.event_start_s + gts.event_length_s + MATCH_TOL_S)).any()
        if hit:
            continue
        frames = range(int(s * fps), int(t_end * fps) + 1)
        c = Counter(NAMES.get(lab_at.get(n), "unlabelled") for n in frames)
        unmatched.append({"video": stem, "kind": e["event"], "start_s": round(s, 2), "length_s": round(e["duration_s"], 2),
                          **{k: c.get(k, 0) for k in ["alert", "yawning", "microsleep", "unlabelled"]}})
        if e["event"] == "long_closure":
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((s + t_end) / 2 * fps))
            ok, fr = cap.read()
            if ok:
                pics.append((f"{stem} {s:.1f}s ({e['duration_s']:.1f}s)\n" +
                             ", ".join(f"{k} {v}" for k, v in c.items()), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()

unmatched = pd.DataFrame(unmatched)
unmatched.to_csv(f"{RESULTS}/alarm_unmatched_closures.csv", index=False)
print("\nSystem closures (>= 0.5 s) with no labelled microsleep; label counts = frames inside the closure:")
display(unmatched)
if pics:
    fig, axes = plt.subplots(1, len(pics), figsize=(5 * len(pics), 3.6))
    for ax, (title, img) in zip(np.atleast_1d(axes), pics):
        ax.imshow(img); ax.set_title(title, fontsize=8); ax.axis("off")
    plt.suptitle("False alarms: middle frame of each closure"); plt.tight_layout()
    plt.savefig(f"{RESULTS}/false_alarms.png", dpi=120, bbox_inches="tight"); plt.show()
