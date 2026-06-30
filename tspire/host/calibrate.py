"""Calibration overlay.

Captures one frame from the game window and draws every named region plus live detection
results (monster HP bars, hand cards) on top, saving an annotated PNG. Use it to tune the
fractional region values in vision/regions.py and the detection thresholds in
vision/backend.py against your real resolution.

Usage:
    python -m tspire.host.calibrate                 # capture live, write calibrate_overlay.png
    python -m tspire.host.calibrate --image shot.png  # annotate a saved screenshot instead
    python -m tspire.host.calibrate --out path.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tspire.host.capture import normalize_frame_to_client
from tspire.host.config import HostConfig
from tspire.host.vision import region_map_for


def _load_frame(args, config: HostConfig):
    import cv2

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
        return _normalize_image_frame(frame, config)
    from tspire.host.capture import WindowCapture

    return WindowCapture(
        config.window_title,
        focus_before_capture=config.focus_before_capture,
    ).grab()


def _normalize_image_frame(frame, config: HostConfig, *, report=print):
    """Crop near-client screenshots that still include a window frame."""
    normalized = normalize_frame_to_client(frame, int(config.width), int(config.height), report=report)
    if normalized is frame and frame.shape[:2] != (int(config.height), int(config.width)):
        h, w = frame.shape[:2]
        report(
            f"warning: image is {w}x{h}; expected {config.width}x{config.height}; "
            "using image dimensions for regions"
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="TSpire region calibration overlay")
    parser.add_argument("--image", help="annotate a saved screenshot instead of capturing")
    parser.add_argument("--out", default="calibrate_overlay.png")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    import cv2

    config = HostConfig.load(args.config)
    frame = _load_frame(args, config)
    h, w = frame.shape[:2]
    regions = region_map_for(w, h)

    from tspire.host.vision.backend import CvVisionBackend

    backend = CvVisionBackend(config.tesseract_cmd)

    overlay = frame.copy()
    # Draw named regions in green with labels.
    for name, rect in regions.all_regions().items():
        left, top, rw, rh = rect.to_pixels(w, h)
        cv2.rectangle(overlay, (left, top), (left + rw, top + rh), (0, 255, 0), 2)
        cv2.putText(overlay, name, (left, max(12, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw detected monster HP bars (red) and hand cards (blue).
    bars = backend.find_red_bars(frame, regions.monster_search)
    for i, b in enumerate(bars):
        cv2.rectangle(overlay, (b.left, b.top), (b.left + b.width, b.top + b.height), (0, 0, 255), 2)
        cv2.putText(overlay, f"m{i}", (b.left, b.top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cards = backend.find_cards(frame, regions.hand_search)
    for i, c in enumerate(cards):
        cv2.rectangle(overlay, (c.left, c.top), (c.left + c.width, c.top + c.height), (255, 128, 0), 2)
        cv2.putText(overlay, f"c{i}", (c.left, c.top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2, cv2.LINE_AA)

    out = Path(args.out)
    cv2.imwrite(str(out), overlay)
    print(f"resolution: {w}x{h}")
    print(f"signals: energy_filled={backend.region_filled(frame, regions.energy)} "
          f"end_turn_filled={backend.region_filled(frame, regions.end_turn)} "
          f"monsters={len(bars)} cards={len(cards)}")
    print(f"wrote overlay -> {out.resolve()}")


if __name__ == "__main__":
    main()
