"""Convert a spacecraft state CSV/TSV to CZML.

Expected columns (header optional; delimiter can be tab, comma, semicolon or spaces):
    time [s since epoch], x, y, z [km, inertial], vx, vy, vz [km/s, ignored]

Optional pointing file (same time base, same units/frame): time, x, y, z. By default each row is a
vector from the spacecraft to the target; use --pointing-mode absolute if rows are target positions.

Example:
    python scripts/csv_to_czml.py data/orbit.tsv public/orbit.czml --epoch 2026-01-01T00:00:00Z --pointing data/pointing.tsv
    python scripts/csv_to_czml.py data/asset1_DynamicStateData.csv public/orbit.czml --epoch 2459295.0 --pointing data/asset1_eci_pointing_vector_xyz_.csv --tasks data/myFirstHSFTaskList.json
"""

import argparse
import json
import re
from bisect import bisect_right
from datetime import datetime, timedelta, timezone


def parse_epoch(text: str) -> datetime:
    try:
        jd = float(text)
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    # JD 2440587.5 is 1970-01-01T00:00:00 UTC
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=jd - 2440587.5)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0").rstrip(".") + "Z"


def read_rows(path: str, scale: float):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p for p in re.split(r"[\t,;\s]+", line.strip()) if p]
            if len(parts) < 4:
                continue
            try:
                t, x, y, z = (float(p) for p in parts[:4])
            except ValueError:
                continue  # header row
            rows.append((t, x * scale, y * scale, z * scale))
    if not rows:
        raise SystemExit(f"No numeric rows found in {path}.")
    return rows


def read_task_targets(path: str):
    """Static ground targets from an HSF task list: stateData is [lat deg, lon deg, alt m]."""
    with open(path, encoding="utf-8-sig") as f:
        tasks = json.load(f)["tasks"]
    targets = []
    for task in tasks:
        target = task.get("target") or {}
        state = target.get("dynamicState") or {}
        if state.get("type") != "STATIC_LLA":
            continue
        lat, lon, alt = (float(v) for v in state["stateData"][:3])
        targets.append(
            {
                "name": target.get("name", task["name"]),
                "task": task["name"],
                "type": target.get("type", ""),
                "value": target.get("value", ""),
                "lla": (lat, lon, alt),
            }
        )
    return targets


def decimate(rows, step: float):
    kept = []
    for row in rows:
        if not kept or row[0] - kept[-1][0] >= step:
            kept.append(row)
    # Keep the final row so the data ends where the input ends.
    if rows[-1][0] > kept[-1][0]:
        kept.append(rows[-1])
    return kept


def interpolate(rows, t: float):
    """Linear interpolation of (t, x, y, z) rows; None outside the covered range."""
    times = [r[0] for r in rows]
    if t < times[0] or t > times[-1]:
        return None
    i = bisect_right(times, t)
    if i == len(rows):
        return rows[-1][1:]
    a, b = rows[i - 1], rows[i]
    f = (t - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
    return tuple(a[k] + f * (b[k] - a[k]) for k in (1, 2, 3))


def pointing_targets(pointing_rows, orbit_rows, mode: str):
    """Return (t, x, y, z) inertial target points in metres."""
    if mode == "absolute":
        return pointing_rows
    targets = []
    for t, dx, dy, dz in pointing_rows:
        sc = interpolate(orbit_rows, t)
        if sc is not None:  # skip pointing rows outside the orbit's time range
            targets.append((t, sc[0] + dx, sc[1] + dy, sc[2] + dz))
    if not targets:
        raise SystemExit("Pointing times do not overlap the orbit times.")
    return targets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--epoch", default="2026-01-01T00:00:00Z", help="UTC time of t=0: ISO 8601 or a Julian date (e.g. 2461041.5)")
    p.add_argument("--name", default="Spacecraft")
    p.add_argument("--step", type=float, default=30.0, help="minimum seconds between kept samples")
    p.add_argument("--units", choices=["km", "m"], default="km")
    p.add_argument("--frame", choices=["INERTIAL", "FIXED"], default="INERTIAL")
    p.add_argument("--tasks", help="HSF task list JSON; each STATIC_LLA target is added as a ground point")
    p.add_argument("--pointing", help="pointing CSV/TSV: time, x, y, z in the same units and inertial frame")
    p.add_argument(
        "--pointing-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="relative: vector from the spacecraft to the target; absolute: target ECI position",
    )
    p.add_argument("--playback-seconds", type=float, default=60.0, help="wall-clock length of the animation")
    args = p.parse_args()

    scale = 1000.0 if args.units == "km" else 1.0
    orbit_rows = read_rows(args.input, scale)
    samples = decimate(orbit_rows, args.step)
    epoch = parse_epoch(args.epoch)
    start = epoch + timedelta(seconds=samples[0][0])
    end = epoch + timedelta(seconds=samples[-1][0])
    interval = f"{iso(start)}/{iso(end)}"
    duration = samples[-1][0] - samples[0][0]

    czml = [
        {
            "id": "document",
            "name": args.name,
            "version": "1.0",
            "clock": {
                "interval": interval,
                "currentTime": iso(start),
                "multiplier": duration / args.playback_seconds if duration > 0 else 1,
                "range": "CLAMPED",
                "step": "SYSTEM_CLOCK_MULTIPLIER",
            },
        },
        {
            "id": "spacecraft",
            "name": args.name,
            "availability": interval,
            "position": {
                "epoch": iso(epoch),
                "referenceFrame": args.frame,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 5,
                "cartesian": [v for s in samples for v in s],
            },
            "point": {
                "pixelSize": 10,
                "color": {"rgba": [255, 255, 0, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
            },
            "label": {
                "text": args.name,
                "font": "14px sans-serif",
                "pixelOffset": {"cartesian2": [0, -20]},
                "fillColor": {"rgba": [255, 255, 255, 255]},
            },
            "path": {
                "show": True,
                "width": 2,
                "leadTime": 0,
                "trailTime": duration,
                "resolution": max(args.step, 30.0),
                "material": {"solidColor": {"color": {"rgba": [255, 255, 0, 255]}}},
            },
        },
    ]

    if args.tasks:
        for target in read_task_targets(args.tasks):
            lat, lon, alt = target["lla"]
            czml.append(
                {
                    "id": f"target_{target['name'].replace(' ', '_')}",
                    "name": target["name"],
                    "description": f"{target['task']} ({target['type']}), value {target['value']}",
                    "position": {"cartographicDegrees": [lon, lat, alt]},
                    "point": {
                        "pixelSize": 8,
                        "color": {"rgba": [0, 255, 128, 255]},
                        "outlineColor": {"rgba": [0, 0, 0, 255]},
                        "outlineWidth": 2,
                        "heightReference": "CLAMP_TO_GROUND" if alt == 0 else "NONE",
                    },
                    "label": {
                        "text": target["name"],
                        "font": "12px sans-serif",
                        "pixelOffset": {"cartesian2": [0, -16]},
                        "fillColor": {"rgba": [255, 255, 255, 255]},
                        "heightReference": "CLAMP_TO_GROUND" if alt == 0 else "NONE",
                    },
                }
            )
        print(f"{len(czml) - 2} targets added from {args.tasks}")

    if args.pointing:
        pointing_rows = read_rows(args.pointing, scale)
        targets = pointing_targets(pointing_rows, orbit_rows, args.pointing_mode)
        p_start = epoch + timedelta(seconds=targets[0][0])
        p_end = epoch + timedelta(seconds=targets[-1][0])
        p_interval = f"{iso(p_start)}/{iso(p_end)}"
        czml.append(
            {
                "id": "pointing_target",
                "name": "Pointing target",
                "availability": p_interval,
                "position": {
                    "epoch": iso(epoch),
                    "referenceFrame": args.frame,
                    "interpolationAlgorithm": "LAGRANGE",
                    "interpolationDegree": 1,
                    "cartesian": [v for s in targets for v in s],
                },
                "point": {"pixelSize": 6, "color": {"rgba": [255, 64, 64, 255]}},
            }
        )
        czml.append(
            {
                "id": "pointing_line",
                "name": "Pointing line",
                "availability": p_interval,
                "polyline": {
                    "positions": {"references": ["spacecraft#position", "pointing_target#position"]},
                    "arcType": "NONE",
                    "width": 2,
                    "material": {"solidColor": {"color": {"rgba": [255, 64, 64, 255]}}},
                },
            }
        )
        print(f"{len(targets)} pointing samples")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(czml, f, separators=(",", ":"))
    print(f"{len(samples)} samples, {duration:.1f} s -> {args.output}")


if __name__ == "__main__":
    main()
