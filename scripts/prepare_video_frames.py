import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a phone video for COLMAP/2DGS.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", default=3.0, type=float)
    parser.add_argument("--max-side", default=1200, type=int)
    return parser.parse_args()


def resize_keep_aspect(frame, max_side):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / args.fps))
    saved = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame = resize_keep_aspect(frame, args.max_side)
            cv2.imwrite(str(args.out / f"{saved:05d}.jpg"), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"Saved {saved} frames to {args.out}")


if __name__ == "__main__":
    main()
