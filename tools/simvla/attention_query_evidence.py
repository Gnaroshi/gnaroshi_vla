"""Inventory completed evidence and sample all saved task videos in selected rows."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.simvla.compile_benchmark import write_json, sha


def video_sheet(path, destination, record):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if count < 1:
        raise RuntimeError(f"Unreadable video: {path}")
    indices = sorted(set(np.linspace(0, count - 1, 12).astype(int).tolist()))
    sheet = Image.new("RGB", (1536, 4 * 282 + 56), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{record['arm']} / task {record['task_id']} / trial {record['trial_id']} / success={record['success']} / actions={record['episode_length']}", fill="black")
    for number, index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Missing frame {index}: {path}")
        tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tile.thumbnail((512, 256))
        x, y = (number % 3) * 512, 56 + (number // 3) * 282
        sheet.paste(tile, (x, y + 22))
        draw.text((x + 6, y + 3), f"frame={index}  before action={2 * index}", fill="black")
    capture.release()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return dict(path=str(path), sha256=sha(path), frames=count, video_fps=fps,
                sampled_frame_indices=indices, sheet=str(destination),
                review_status="extracted_not_yet_visually_reviewed", **record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    args = parser.parse_args()
    inventory = []
    for root in args.root:
        for summary_path in sorted(root.glob("rows/*/summary.json")):
            summary = json.loads(summary_path.read_text())
            arm = summary_path.parent.name
            records = []
            for p in sorted(root.glob(f"shards/{arm}/part*/rows/{arm}/episodes/*.json")):
                record = json.loads(p.read_text())
                actions_path = p.with_suffix(".npz")
                if actions_path.exists():
                    actions = np.load(actions_path)["actions"]
                    delta = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
                    record["translation_command_delta_p95"] = float(np.quantile(delta, .95)) if len(delta) else 0.
                records.append(record)
            videos = []
            for path in sorted(root.glob(f"shards/{arm}/part*/rows/{arm}/videos/*.mp4")):
                task = int(path.stem.split("_")[0][4:])
                trial = int(path.stem.split("_")[1][5:])
                record = next(r for r in records if (r["task_id"], r["trial_id"]) == (task, trial))
                destination = args.output / "video_sheets" / root.parent.name / arm / (path.stem + ".jpg")
                videos.append(video_sheet(path, destination, record))
            inventory.append(dict(root=str(root), summary_path=str(summary_path), summary=summary,
                                  episode_records=records, videos=videos))
    write_json(args.output / "evidence.json", dict(rows=inventory,
        note="Frames cover every saved video in these rows, not every evaluated episode. Contact sheets do not establish first physical cause."))
    print(json.dumps(dict(rows=len(inventory), episodes=sum(len(r['episode_records']) for r in inventory),
                          videos=sum(len(r['videos']) for r in inventory))))


if __name__ == "__main__":
    main()
