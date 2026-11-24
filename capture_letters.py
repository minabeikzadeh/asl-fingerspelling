"""
Webcam photo capture for the team's own ASL test set.

Press a letter key while signing that letter -> the photo is saved to
    our_recordings/<LETTER>/<name>_<room>_<number>.jpg
which is exactly the folder layout extract_keypoints.py expects.

Usage (one run per room / lighting setup):
    pip install opencv-python
    python capture_letters.py --name mina --room room1
    python capture_letters.py --name mina --room hallway

Keys:
    a-y (not j, z)  save a photo for that letter
    Backspace       delete the last photo you saved (undo)
    Esc             quit

Tips:
- Keep your signing hand inside the green square; only the square is saved,
  so your face stays out of the photos.
- Use your RIGHT hand, like the Kaggle signer.
- Change the hand's angle/distance a little between shots.
- Don't skip photos because a hand "looks hard": the test set should include
  hard cases, or the results will look better than they really are.
"""
import argparse
from pathlib import Path

import cv2

LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]
ESC, BACKSPACE = 27, 8


def key_to_letter(key):
    """Map a key code to one of the 24 letters, or None."""
    if key < 0:
        return None
    ch = chr(key & 0xFF).upper()
    return ch if ch in LETTERS else None


def square_box(frame_w, frame_h, frac=0.7):
    """Centered square guide box covering `frac` of the frame height."""
    side = int(min(frame_w, frame_h) * frac)
    x0, y0 = (frame_w - side) // 2, (frame_h - side) // 2
    return x0, y0, side


def next_path(out_dir, letter, name, room):
    """Next free file name for this person/room/letter (so you can stop and resume)."""
    folder = Path(out_dir) / letter
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while (folder / f"{name}_{room}_{n:02d}.jpg").exists():
        n += 1
    return folder / f"{name}_{room}_{n:02d}.jpg"


def count_saved(out_dir, letter, name, room):
    folder = Path(out_dir) / letter
    return len(list(folder.glob(f"{name}_{room}_*.jpg"))) if folder.is_dir() else 0


def open_camera(index):
    # On Windows, the DirectShow backend opens the webcam much faster
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --camera 1, or close "
                         f"other apps using the webcam (Zoom, Teams, browser tabs).")
    return cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="your name, e.g. mina")
    p.add_argument("--room", required=True, help="room/lighting tag, e.g. room1, hallway, dim")
    p.add_argument("--out", default="our_recordings", help="output folder")
    p.add_argument("--target", type=int, default=5, help="photos per letter for this room")
    p.add_argument("--camera", type=int, default=0, help="webcam number (try 1 if 0 fails)")
    args = p.parse_args()

    name = args.name.strip().lower().replace(" ", "")
    room = args.room.strip().lower().replace(" ", "")
    cap = open_camera(args.camera)
    counts = {l: count_saved(args.out, l, name, room) for l in LETTERS}
    last_saved = None
    message = "Sign a letter, then press its key"
    print(f"Saving to {Path(args.out).resolve()}  |  Esc to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break
        h, w = frame.shape[:2]
        x0, y0, side = square_box(w, h)

        # What you see is mirrored like a selfie, but the SAVED photo is the
        # camera's real (unmirrored) view, matching how the Kaggle photos were taken.
        view = cv2.flip(frame, 1)
        vx0 = w - x0 - side
        cv2.rectangle(view, (vx0, y0), (vx0 + side, y0 + side), (0, 200, 0), 2)
        cv2.putText(view, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        done = sum(min(c, args.target) for c in counts.values())
        todo = [l for l in LETTERS if counts[l] < args.target]
        cv2.putText(view, f"{name}/{room}: {done}/{len(LETTERS) * args.target}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(view, "Still need: " + ("".join(todo) if todo else "all done!"),
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("ASL capture (Esc to quit)", view)

        key = cv2.waitKey(1)
        if key == ESC:
            break
        if key == BACKSPACE and last_saved is not None:
            path, letter = last_saved
            if path.exists():
                path.unlink()
                counts[letter] -= 1
                message = f"Deleted {path.name}"
            last_saved = None
            continue
        letter = key_to_letter(key)
        if letter:
            crop = frame[y0:y0 + side, x0:x0 + side]      # only the green square
            path = next_path(args.out, letter, name, room)
            cv2.imwrite(str(path), crop)
            counts[letter] += 1
            last_saved = (path, letter)
            message = f"Saved {letter} ({counts[letter]}/{args.target})"
        elif key not in (-1, 255) and chr(key & 0xFF).upper() in "JZ":
            message = "J and Z are skipped (they need motion)"

    cap.release()
    cv2.destroyAllWindows()
    total = sum(counts.values())
    missing = [l for l in LETTERS if counts[l] < args.target]
    print(f"Done. {total} photos for {name}/{room}.")
    if missing:
        print("Letters below target:", " ".join(f"{l}({counts[l]})" for l in missing))


if __name__ == "__main__":
    main()
