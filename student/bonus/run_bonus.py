"""Bonus experiments for the Day 23 lab (RUBRIC section 2).

Runs outside ``platform/`` and never edits it. One detection pass over the segment is
cached, then the student tracker is replayed several times:

* calibration study (+4): camera extrinsic yaw is rotated by several offsets; we log the
  camera innovation, chi-square gate pass rate and RMSE per offset;
* visualisation (+3): annotated BEV, error-over-time and camera overlay figures;
* CVAT hook (+3, partial): per-frame confirmed tracks exported with
  ``fusion_lab.export_cvat.export_tracks_json`` (the CVAT import itself is manual).

Usage (repo root, after ``fusion_lab`` is importable)::

    python student/bonus/run_bonus.py --config student/config/paths.yaml

Outputs are written to ``student/bonus/``. ``student/artifacts/`` is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BONUS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("DAY23_STUDENT_ROOT", str(BONUS_DIR.parent))

from fusion_lab import tracking_params  # noqa: E402
from fusion_lab.evaluation import TRACK_GATE_METERS, tracking_counts  # noqa: E402
from fusion_lab.export_cvat import export_tracks_json  # noqa: E402
from fusion_lab.tracking.filter import Filter  # noqa: E402
from fusion_lab.tracking.manager import TrackManager  # noqa: E402
from fusion_lab.tracking.sensors import Sensor  # noqa: E402
from fusion_lab.workspace_loader import load_workspace  # noqa: E402

DEFAULT_YAWS = (0.0, 1.0, 2.0, 5.0, 10.0)


@dataclass
class FrameData:
    cnt: int
    detections: Any
    labels: list
    cam_centres: list | None
    cam_boxes: list | None
    image: bytes | None


@dataclass
class Cache:
    frames: list[FrameData]
    det_cfg: Any
    cam_calib: Any
    segment: str


@dataclass
class RunResult:
    records: list[dict] = field(default_factory=list)
    tracks_by_frame: dict[int, list[dict]] = field(default_factory=dict)
    cam_events: list[dict] = field(default_factory=list)
    gate_pairs: int = 0
    gate_passed: int = 0

    @property
    def matches(self) -> int:
        return sum(r["matches"] for r in self.records)

    @property
    def sum_sq_err(self) -> float:
        return float(sum(r["sum_sq_err"] for r in self.records))

    @property
    def rmse(self) -> float | None:
        return float(np.sqrt(self.sum_sq_err / self.matches)) if self.matches else None


# ----------------------------------------------------------------------------- data
def load_cache(config_path: Path, max_frames: int | None = None):
    """Run the (shared) detector once and keep what the tracker replays need."""
    import torch
    from fusion_lab.evaluation import valid_ground_truth
    from fusion_lab.lidar_pcl import pcl_from_range_image
    from fusion_lab.scripts import run_lab as rl

    rl._setup_import_paths()
    cfg = rl._load_paths_config(config_path)
    ws = load_workspace()
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    segment = cfg.get("segment")
    frame_start = int(cfg.get("frame_start", 0))
    frame_end = int(cfg.get("frame_end", 20))
    if max_frames is not None:
        frame_end = min(frame_end, frame_start + max_frames - 1)
    weights = rl._resolve_weights(cfg)
    det_cfg = ws["detection_pipeline"].load_fpn_resnet_config(str(weights) if weights else None)
    model = ws["detection_pipeline"].create_fpn_model(det_cfg, str(weights) if weights else None)
    reader = WaymoDataFileReader(str(Path(cfg["waymo_dir"]) / segment))
    bev = ws["bev_mapping"]

    frames, cam_calib = [], None
    for cnt, frame in enumerate(iter(reader)):
        if cnt < frame_start:
            continue
        if cnt > frame_end:
            break
        if cam_calib is None:
            cam_calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = ws["detection_pipeline"].detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)
        group = next((g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT), None)
        centres = boxes = None
        if group is not None:
            veh = [l for l in group.labels if l.type == label_pb2.Label.Type.TYPE_VEHICLE]
            centres = [(l.box.center_x, l.box.center_y) for l in veh]
            boxes = [(l.box.center_x, l.box.center_y, l.box.length, l.box.width) for l in veh]
        img = next((im.image for im in frame.images if im.name == dataset_pb2.CameraName.FRONT), None)
        frames.append(FrameData(cnt, detections, list(labels), centres, boxes, img))
        print(f"[cache] frame {cnt}/{frame_end}", end="\r")
    print()
    return ws, Cache(frames, det_cfg, cam_calib, segment)


# -------------------------------------------------------------------------- tracker
def rotate_extrinsic(sensor: Sensor, yaw_deg: float) -> None:
    """Rotate the camera mount about its own vertical axis (a calibration error)."""
    a = np.deg2rad(yaw_deg)
    rz = np.eye(4)
    rz[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    new = np.asarray(sensor.sens_to_veh, dtype=float) @ rz
    sensor.sens_to_veh = np.asmatrix(new)
    sensor.veh_to_sens = np.asmatrix(np.linalg.inv(new))


class SpyFilter(Filter):
    """Filter that logs the camera innovation before each camera update."""

    def __init__(self, kalman_mod, sink: list):
        super().__init__(kalman_mod)
        self.sink, self.frame, self.labels = sink, -1, []

    def _err_to_gt(self, pos):
        if not self.labels:
            return None
        centres = np.array([(l.box.center_x, l.box.center_y, l.box.center_z) for l in self.labels])
        d_xy = np.linalg.norm(centres[:, :2] - pos[:2], axis=1)
        j = int(np.argmin(d_xy))
        return float(np.linalg.norm(centres[j] - pos)) if d_xy[j] <= TRACK_GATE_METERS else None

    def update(self, track, meas):
        if meas.sensor.name != "camera":
            return super().update(track, meas)
        sensor = meas.sensor
        gamma = np.asarray(self._k.innovation(track.x, meas), dtype=float).reshape(-1)
        S = np.asarray(self._k.innovation_covariance(track.P, meas, sensor.get_H(track.x)), dtype=float)
        d2 = float(gamma @ np.linalg.inv(S) @ gamma)
        pre_px = np.asarray(sensor.get_hx(track.x), dtype=float).reshape(-1)
        before = np.asarray(track.x, dtype=float).reshape(-1)[:3].copy()
        super().update(track, meas)
        after = np.asarray(track.x, dtype=float).reshape(-1)[:3].copy()
        post_px = np.asarray(sensor.get_hx(track.x), dtype=float).reshape(-1)
        self.sink.append({
            "frame": self.frame, "track_id": int(track.id), "state": track.state,
            "innovation_px": float(np.linalg.norm(gamma)), "d2": d2,
            "meas_px": np.asarray(meas.z, dtype=float).reshape(-1).tolist(),
            "pre_px": pre_px.tolist(), "post_px": post_px.tolist(),
            "pos_before": before.tolist(), "pos_after": after.tolist(),
            "err_before": self._err_to_gt(before), "err_after": self._err_to_gt(after),
        })


def run_tracker(ws, cache: Cache, mode: str, seed: int = 0, yaw_deg: float = 0.0) -> RunResult:
    """Replay the student tracker on cached detections (same loop as run_lab.run)."""
    from fusion_lab.scripts import run_lab as rl

    assoc, kalman, cam = ws["association"], ws["kalman"], ws["camera_fusion"]
    result = RunResult()
    rng = np.random.default_rng(seed)
    kf = SpyFilter(kalman, result.cam_events)
    manager = TrackManager(ws["track_management"])
    lidar = Sensor("lidar", None, cam)
    camera = None
    if mode == "fused":
        camera = Sensor("camera", cache.cam_calib, cam)
        if yaw_deg:
            rotate_extrinsic(camera, yaw_deg)

    original_gate = assoc.chi2_gate

    def counting_gate(mhd_sq, sensor):
        passed = original_gate(mhd_sq, sensor)
        if sensor.name == "camera":
            result.gate_pairs += 1
            result.gate_passed += int(passed)
        return passed

    assoc.chi2_gate = counting_gate
    try:
        for item in cache.frames:
            observations = rl._lidar_observations(item.cnt, item.detections, lidar, cache.det_cfg)
            for track in manager.track_list:
                kf.predict(track)
                track.set_t(item.cnt * tracking_params.dt)
            kf.frame, kf.labels = item.cnt, item.labels
            assoc.associate_and_update(manager, observations, kf, lidar)
            if camera is not None and item.cam_centres is not None:
                cam_obs = []
                for centre in item.cam_centres:
                    camera.generate_measurement(item.cnt, np.array(centre) + rng.normal(0, 0.5, 2), cam_obs)
                assoc.associate_and_update(manager, cam_obs, kf, camera)
            counts = tracking_counts(manager.track_list, item.labels)
            result.records.append({"frame": item.cnt, **counts})
            result.tracks_by_frame[item.cnt] = [
                {"id": int(t.id), "x": float(t.x[0, 0]), "y": float(t.x[1, 0]), "z": float(t.x[2, 0]),
                 "height": float(t.height), "width": float(t.width), "length": float(t.length),
                 "yaw": float(t.yaw), "score": float(t.score), "state": t.state}
                for t in manager.track_list if t.state == "confirmed"
            ]
    finally:
        assoc.chi2_gate = original_gate
    return result


# --------------------------------------------------------------------- calibration
def summarize(result: RunResult, lidar_rmse: float | None, yaw: float | None) -> dict:
    ev = result.cam_events
    inn = np.array([e["innovation_px"] for e in ev]) if ev else np.array([])
    d2 = np.array([e["d2"] for e in ev]) if ev else np.array([])
    pairs = [(e["err_before"], e["err_after"]) for e in ev
             if e["err_before"] is not None and e["err_after"] is not None]
    rmse = result.rmse
    return {
        "yaw_deg": yaw, "rmse": rmse,
        "delta_vs_lidar": None if rmse is None or lidar_rmse is None else rmse - lidar_rmse,
        "matches": result.matches,
        "ghosts": sum(r["ghosts"] for r in result.records),
        "misses": sum(r["misses"] for r in result.records),
        "camera_pairs_in_fov": result.gate_pairs, "camera_pairs_passed_gate": result.gate_passed,
        "gate_pass_rate": result.gate_passed / result.gate_pairs if result.gate_pairs else None,
        "camera_updates": len(ev),
        "innovation_px_median": float(np.median(inn)) if inn.size else None,
        "innovation_px_mean": float(np.mean(inn)) if inn.size else None,
        "d2_mean": float(np.mean(d2)) if d2.size else None,
        "err_before_update_mean": float(np.mean([p[0] for p in pairs])) if pairs else None,
        "err_after_update_mean": float(np.mean([p[1] for p in pairs])) if pairs else None,
        "n_err_pairs": len(pairs),
    }


def fmt(v, nd=3, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def calibration_table(rows: list[dict]) -> str:
    head = ("| Lệch yaw extrinsic (độ) | RMSE (m) | RMSE − LiDAR (m) | số cặp ghép | số lần update camera | "
            "tỉ lệ qua cổng χ² | median \\|γ\\| (px) | mean d² | sai số vị trí trước → sau update camera (m) |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        tag = "LiDAR only" if r["yaw_deg"] is None else f"{r['yaw_deg']:g}"
        eb, ea = r["err_before_update_mean"], r["err_after_update_mean"]
        ba = "n/a" if eb is None else f"{eb:.3f} → {ea:.3f}"
        lines.append(
            f"| {tag} | {fmt(r['rmse'])} | {fmt(r['delta_vs_lidar'])} | {r['matches']} | {r['camera_updates']} | "
            f"{fmt(r['gate_pass_rate'], pct=True)} | {fmt(r['innovation_px_median'], 1)} | "
            f"{fmt(r['d2_mean'], 1)} | {ba} |")
    return head + "\n".join(lines) + "\n"


def run_calibration(ws, cache, seed, yaws, lidar: RunResult):
    rows = [summarize(lidar, lidar.rmse, None)]
    runs = {}
    for yaw in yaws:
        res = run_tracker(ws, cache, "fused", seed, yaw)
        runs[yaw] = res
        rows.append(summarize(res, lidar.rmse, yaw))
        print(f"[calib] yaw={yaw:g} deg  rmse={fmt(res.rmse)}  gate={fmt(rows[-1]['gate_pass_rate'], pct=True)}")
    return rows, runs


# ------------------------------------------------------------------- visualisation
def _nearest_gt(labels, x, y):
    if not labels:
        return None
    c = np.array([(l.box.center_x, l.box.center_y, l.box.center_z) for l in labels])
    d = np.linalg.norm(c[:, :2] - np.array([x, y]), axis=1)
    j = int(np.argmin(d))
    return (j, c[j]) if d[j] <= TRACK_GATE_METERS else None


def pick_event(events: list[dict]):
    """Camera update on a confirmed track with the largest error reduction."""
    usable = [e for e in events if e["state"] == "confirmed" and e["err_before"] is not None
              and e["err_after"] is not None]
    if not usable:
        return None
    return max(usable, key=lambda e: e["err_before"] - e["err_after"])


def fig_bev(cache, lidar, fused, frame, event, path):
    item = next(f for f in cache.frames if f.cnt == frame)
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    gt = np.array([(l.box.center_x, l.box.center_y, l.box.center_z) for l in item.labels]).reshape(-1, 3)
    if len(gt):
        ax.scatter(gt[:, 1], gt[:, 0], s=220, facecolors="none", edgecolors="green", linewidths=2, label="Ground truth (tâm xe)")
    det = np.asarray(item.detections, dtype=float).reshape(-1, 8) if len(item.detections) else np.zeros((0, 8))
    if len(det):
        ax.scatter(det[:, 2], det[:, 1], marker="+", s=120, c="tab:blue", label="Detection LiDAR (đo)")
    lt, ft = lidar.tracks_by_frame[frame], fused.tracks_by_frame[frame]
    if lt:
        ax.scatter([t["y"] for t in lt], [t["x"] for t in lt], s=70, c="orange", edgecolors="k", label="Track — chỉ LiDAR")
    if ft:
        ax.scatter([t["y"] for t in ft], [t["x"] for t in ft], marker="x", s=90, c="red", label="Track — LiDAR + camera")
    lines = []
    for t in ft:
        g = _nearest_gt(item.labels, t["x"], t["y"])
        if g is None:
            continue
        ef = float(np.linalg.norm(np.array([t["x"], t["y"], t["z"]]) - g[1]))
        tl = min(lt, key=lambda u: (u["x"] - t["x"]) ** 2 + (u["y"] - t["y"]) ** 2) if lt else None
        el = float(np.linalg.norm(np.array([tl["x"], tl["y"], tl["z"]]) - g[1])) if tl else float("nan")
        lines.append(f"ID {t['id']}: lỗi 3D LiDAR {el:.3f} m → fused {ef:.3f} m")
        ax.annotate(f"ID {t['id']}", (t["y"], t["x"]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    if event is not None and event["frame"] == frame:
        lines.append(f"Camera update (track {event['track_id']}): sai số {event['err_before']:.3f} → {event['err_after']:.3f} m")
    ax.text(0.02, 0.02, "\n".join(lines) or "không có track confirmed ghép GT", transform=ax.transAxes,
            fontsize=9, va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    pts = [(t["y"], t["x"]) for t in lt + ft] + [(g[1], g[0]) for g in gt] + [(d[2], d[1]) for d in det]
    if pts:
        arr = np.array(pts)
        pad = 4.0
        ax.set_xlim(arr[:, 0].max() + pad, arr[:, 0].min() - pad)  # inverted: left (y>0) on the left
        ax.set_ylim(arr[:, 1].min() - pad, arr[:, 1].max() + pad)
    ax.set_xlabel("y (m, trái dương)")
    ax.set_ylabel("x (m, phía trước)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    ax.set_title(f"BEV frame {frame}: GT, detection, track LiDAR-only vs fused")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_error_over_time(lidar, fused, frame, path):
    def per_frame(res):
        xs, ys = [], []
        for r in res.records:
            if r["matches"]:
                xs.append(r["frame"])
                ys.append(np.sqrt(r["sum_sq_err"] / r["matches"]))
        return xs, ys

    fig, ax = plt.subplots(figsize=(9, 4))
    xl, yl = per_frame(lidar)
    xf, yf = per_frame(fused)
    ax.plot(xl, yl, color="orange", label=f"LiDAR only (RMSE {fmt(lidar.rmse)} m)")
    ax.plot(xf, yf, color="red", label=f"LiDAR + camera (RMSE {fmt(fused.rmse)} m)")
    if frame is not None:
        ax.axvline(frame, color="gray", ls="--", lw=1)
        ax.text(frame, ax.get_ylim()[1] * 0.95, f" frame {frame}", fontsize=8, va="top")
    ax.set_xlabel("frame")
    ax.set_ylabel("sai số vị trí 3D trung bình theo frame (m)")
    ax.set_title("Sai số tracking theo thời gian (confirmed track ghép GT)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_camera_overlay(cache, event, path):
    import cv2

    item = next(f for f in cache.frames if f.cnt == event["frame"])
    if item.image is None:
        return False
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(item.image, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    pre, post, meas = (np.array(event[k]) for k in ("pre_px", "post_px", "meas_px"))
    for ax in axes:
        ax.imshow(img)
        for (cx, cy, length, width) in item.cam_boxes or []:
            ax.add_patch(plt.Rectangle((cx - length / 2, cy - width / 2), length, width, fill=False, ec="cyan", lw=1.2))
        ax.scatter(*meas, marker="o", s=90, facecolors="none", edgecolors="magenta", linewidths=2, label="đo camera (tâm hộp 2D GT + nhiễu)")
        ax.scatter(*pre, marker="x", s=90, c="yellow", linewidths=2.5, label="chiếu track TRƯỚC update camera")
        ax.scatter(*post, marker="+", s=140, c="lime", linewidths=2.5, label="chiếu track SAU update camera")
    span = max(60.0, 3 * max(np.linalg.norm(pre - meas), np.linalg.norm(post - meas)))
    axes[1].set_xlim(meas[0] - span, meas[0] + span)
    axes[1].set_ylim(meas[1] + span, meas[1] - span)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_title(f"Camera FRONT, frame {event['frame']} (hộp cyan = nhãn 2D GT)")
    axes[1].set_title("Phóng to quanh điểm đo")
    fig.suptitle(
        f"Track {event['track_id']}: khoảng cách pixel tới đo {np.linalg.norm(pre - meas):.1f} → {np.linalg.norm(post - meas):.1f} px; "
        f"sai số vị trí 3D {event['err_before']:.3f} → {event['err_after']:.3f} m", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def fig_calibration(rows, path):
    data = [r for r in rows if r["yaw_deg"] is not None]
    yaws = [r["yaw_deg"] for r in data]
    lidar_rmse = rows[0]["rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(yaws, [r["innovation_px_median"] or np.nan for r in data], "o-")
    axes[0].set_ylabel("median |γ| (px)")
    axes[0].set_title("Innovation camera")
    axes[1].plot(yaws, [100 * (r["gate_pass_rate"] or 0) for r in data], "o-", c="tab:purple")
    axes[1].set_ylabel("tỉ lệ cặp qua cổng χ² (%)")
    axes[1].set_title("Gating")
    axes[2].plot(yaws, [r["rmse"] or np.nan for r in data], "o-", c="red", label="fused")
    if lidar_rmse is not None:
        axes[2].axhline(lidar_rmse, color="orange", ls="--", label="LiDAR only")
    axes[2].set_ylabel("RMSE (m)")
    axes[2].set_title("RMSE")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("lệch yaw extrinsic camera (độ)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def run_all(ws, cache, out: Path, seed: int = 0, yaws=DEFAULT_YAWS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    lidar = run_tracker(ws, cache, "lidar", seed)
    rows, runs = run_calibration(ws, cache, seed, yaws, lidar)
    base = runs[0.0] if 0.0 in runs else run_tracker(ws, cache, "fused", seed, 0.0)

    (out / "calibration_results.json").write_text(json.dumps(rows, indent=2))
    (out / "calibration_table.md").write_text(calibration_table(rows), encoding="utf-8")

    event = pick_event(base.cam_events)
    frame = event["frame"] if event else cache.frames[len(cache.frames) // 2].cnt
    fig_bev(cache, lidar, base, frame, event, out / "fig1_bev_lidar_vs_fused.png")
    fig_error_over_time(lidar, base, frame, out / "fig2_error_over_time.png")
    overlay = bool(event) and fig_camera_overlay(cache, event, out / "fig3_camera_overlay.png")
    fig_calibration(rows, out / "fig4_calibration_study.png")

    export_tracks_json(
        [{"frame": c, "tracks": base.tracks_by_frame[c]} for c in sorted(base.tracks_by_frame)],
        out / "tracks_fused_cvat.json",
    )
    all_pairs = [(e["err_before"], e["err_after"]) for e in base.cam_events
                 if e["err_before"] is not None and e["err_after"] is not None]
    summary = {
        "frames": [cache.frames[0].cnt, cache.frames[-1].cnt], "seed": seed, "segment": cache.segment,
        "lidar_rmse": lidar.rmse, "fused_rmse_no_offset": base.rmse,
        "baseline_camera_updates": len(base.cam_events),
        "camera_update_mean_err_before": float(np.mean([p[0] for p in all_pairs])) if all_pairs else None,
        "camera_update_mean_err_after": float(np.mean([p[1] for p in all_pairs])) if all_pairs else None,
        "camera_update_share_improved": float(np.mean([p[1] < p[0] for p in all_pairs])) if all_pairs else None,
        "highlight_event": event, "overlay_saved": overlay,
        "figures": [p.name for p in sorted(out.glob("fig*.png"))],
    }
    (out / "bonus_summary.json").write_text(json.dumps(summary, indent=2))
    print(calibration_table(rows))
    print(json.dumps({k: v for k, v in summary.items() if k != "highlight_event"}, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Day 23 bonus experiments")
    parser.add_argument("--config", type=Path, default=BONUS_DIR.parent / "config" / "paths.yaml")
    parser.add_argument("--out", type=Path, default=BONUS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--yaws", type=float, nargs="+", default=list(DEFAULT_YAWS))
    args = parser.parse_args(argv)
    ws, cache = load_cache(args.config, args.max_frames)
    run_all(ws, cache, args.out, args.seed, tuple(args.yaws))


if __name__ == "__main__":
    main()
