from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

PIANOVAM_PATH = Path("DataSets/PianoVAM_v1.1/MIDI")
MAESTRO_ROOT = Path("DataSets/Maestro/maestro-v3.0.0")
OUTPUT_DIR = Path("Data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 22012681


def extract_notes(midi_path):
    """
    Parse a MIDI file into a sorted list of note dicts.

    Each dict contains:
        - onset    : time in seconds when the note starts
        - duration : how long the key is held (seconds)
        - velocity : strike force (0–127)
        - pitch    : 0
        - ioi      : time since previous note onset (0.0 for first note)

    Returns an empty list if the file cannot be parsed.
    """

    res = []
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"[WARN] Skipping {midi_path}: {e}")
        return res

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            res.append(
                {
                    "onset": note.start,
                    "duration": note.end - note.start,
                    "velocity": note.velocity,
                    "pitch": 0,
                    "ioi": 0.0,
                }
            )

        res.sort(key=lambda n: n["onset"])

        for i in range(1, len(res)):
            res[i]["ioi"] = res[i]["onset"] - res[i - 1]["onset"]

    return res


def segment_notes(notes, seg_len=45, hop=22):
    """
    Slice a note list into overlapping fixed-length time windows.

    Args:
        notes   : sorted list of note dicts from extract_notes()
        seg_len : window size in seconds (default 45)
        hop     : step size in seconds (default 22, ~50% overlap)

    Returns a list of segments, where each segment is a list of note dicts
    whose onset falls within [start, start + seg_len).
    Segments with fewer than 20 notes are discarded.
    The first note of each segment has its IOI reset to 0.0.
    """

    if not notes:
        return []

    last = notes[-1]["onset"]
    res = []
    start = 0.0

    while start <= last:
        temp = [n for n in notes if start <= n["onset"] < start + seg_len]

        if len(temp) >= 20:
            temp[0]["ioi"] = 0.0
            res.append(temp)

        start += hop

    return res


def notes_to_array(notes, max_notes=512):
    """
    Convert a list of note dicts into a fixed-size float32 numpy array.

    Each row represents one note with 4 globally normalised features:
        col 0 — onset    : relative position in segment (0.0 → 1.0)
        col 1 — duration : key hold time, capped at 4s   (0.0 → 1.0)
        col 2 — velocity : strike force / 127            (0.0 → 1.0)
        col 3 — ioi      : inter-onset interval, capped at 2s (0.0 → 1.0)

    Notes beyond max_notes are truncated.
    Rows beyond the actual note count are zero-padded.

    Returns array of shape (max_notes, 4), dtype float32.
    """

    res = np.zeros((max_notes, 4), dtype=np.float32)
    seg_start = notes[0]["onset"]
    seg_len = max(1.0, notes[-1]["onset"] - seg_start)
    for i in range(min(len(notes), max_notes)):
        res[i][0] = (notes[i]["onset"] - seg_start) / seg_len
        res[i][1] = min(notes[i]["duration"], 4.0) / 4.0
        res[i][2] = notes[i]["velocity"] / 127
        res[i][3] = min(notes[i]["ioi"], 2.0) / 2.0

    return res


def main():
    """
    Full preprocessing pipeline for both datasets.

    For each MIDI file:
        1. Extract notes (pitch ignored)
        2. Slice into overlapping 45s segments (22s hop)
        3. Convert each segment to a (512, 4) float32 numpy array
        4. Save each array as a .npy file in OUTPUT_DIR

    Then builds a balanced manifest.csv with equal amateur/virtuoso
    segments, shuffled and split into train/val/test (70/15/15).
    """

    pv_files = sorted(PIANOVAM_PATH.glob("*.mid"))
    mae_files = sorted(MAESTRO_ROOT.rglob("*.midi"))
    records = []

    for p in pv_files:
        notes = extract_notes(p)
        segs = segment_notes(notes)
        for seg_idx, seg in enumerate(segs):
            arr = notes_to_array(seg)
            out_name = f"amateur_{p.stem}_seg{seg_idx:03d}.npy"
            out_path = OUTPUT_DIR / out_name
            np.save(out_path, arr)
            records.append(
                {
                    "file": str(out_path),
                    "label": 0,
                    "source": p.name,
                    "seg_idx": seg_idx,
                    "n_notes": len(seg),
                }
            )

    for p in mae_files:
        notes = extract_notes(p)
        segs = segment_notes(notes)
        for seg_idx, seg in enumerate(segs):
            arr = notes_to_array(seg)
            out_name = f"virtuoso_{p.stem}_seg{seg_idx:03d}.npy"
            out_path = OUTPUT_DIR / out_name
            np.save(out_path, arr)
            records.append(
                {
                    "file": str(out_path),
                    "label": 1,
                    "source": p.name,
                    "seg_idx": seg_idx,
                    "n_notes": len(seg),
                }
            )

    df = pd.DataFrame(records)
    df_amateur = df[df["label"] == 0]
    df_virtuoso = df[df["label"] == 1]

    print(f"Amateur  segments: {len(df_amateur)}")
    print(f"Virtuoso segments: {len(df_virtuoso)}")

    df_size = min(len(df_amateur), len(df_virtuoso))

    df_amateur = df_amateur.sample(df_size, random_state=RANDOM_SEED)
    df_virtuoso = df_virtuoso.sample(df_size, random_state=RANDOM_SEED)

    df_balanced = (
        pd.concat([df_amateur, df_virtuoso])
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    n = len(df_balanced)
    df_balanced["split"] = "train"
    df_balanced.loc[int(n * 0.70) : int(n * 0.85), "split"] = "val"
    df_balanced.loc[int(n * 0.85) :, "split"] = "test"

    df_balanced.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    print(f"\nDONE ")
    print(f"  Amateur  segments : {df_size}")
    print(f"  Virtuoso segments : {df_size}")
    print(f"  Total balanced    : {len(df_balanced)}")
    print(df_balanced["split"].value_counts().to_string())
    print(f"  Manifest saved → {OUTPUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    pv_file = next(Path("DataSets/PianoVAM_v1.1/MIDI/").glob("*.mid"))
    notes = extract_notes(pv_file)

    print(f"Total notes : {len(notes)}")
    print(f"First note  : {notes[0]}")
    print(f"Second note : {notes[1]}")
    print(f"IOI[0] should be 0.0 : {notes[0]['ioi']}")
    print(f"All pitches 0        : {all(n['pitch'] == 0 for n in notes)}")
    print(f"Any negative IOI     : {any(n['ioi'] < 0 for n in notes)}")

    segments = segment_notes(notes, seg_len=45, hop=22)

    print(f"Number of segments     : {len(segments)}")
    print(f"Notes in segment 0     : {len(segments[0])}")
    print(f"Notes in segment 1     : {len(segments[1])}")
    print(f"First note IOI (seg 0) : {segments[0][0]['ioi']}")
    print(f"First note IOI (seg 1) : {segments[1][0]['ioi']}")
    print(
        f"Seg 0 onset range      : {segments[0][0]['onset']:.1f}s → {segments[0][-1]['onset']:.1f}s"
    )
    print(
        f"Seg 1 onset range      : {segments[1][0]['onset']:.1f}s → {segments[1][-1]['onset']:.1f}s"
    )

    arr = notes_to_array(
        segments[0][1] if isinstance(segments[0], tuple) else segments[0]
    )

    print(f"Array shape        : {arr.shape}")
    print(f"Array dtype        : {arr.dtype}")
    print(f"First row          : {arr[0]}")
    print(f"Second row         : {arr[1]}")
    print(f"Last row (padding) : {arr[-1]}")
    print(f"All values 0–1     : {arr.min() >= 0.0 and arr.max() <= 1.0}")
    print(f"Non-zero rows      : {(arr.sum(axis=1) != 0).sum()}")

    main()
