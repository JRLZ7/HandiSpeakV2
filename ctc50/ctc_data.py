# ctc_data_test.py
# Fixed 10-word synthetic sentences from per-word JSON bundles (your exact format).
# Compatible with nn.CTCLoss (blank=0). Feature order matches data_keypoints.py.

import os
import json
import random
from glob import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# === Landmark indices copied to preserve EXACT order used in your extractor ===
FACE_LANDMARKS = [
    # Left eyebrow (10)
    70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
    # Right eyebrow (10)
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
    # Left eye (16)
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Right eye (16)
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Upper lip (11)
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # Lower lip (9)
    95, 88, 178, 87, 14, 317, 402, 318, 324
    # 468, 473 intentionally absent (iris) to match your extractor.
]
POSE_LANDMARKS = [11, 12, 13, 14]   # Upper torso/arms — exactly as in your extractor.
HAND_LANDMARK_COUNT = 21            # Left & right hand indices 0..20

# Total landmarks per frame = 72 (face) + 4 (pose) + 21 (LH) + 21 (RH) = 118
# Features per frame = 118 * 3 (x,y,z) = 354
FEATURE_DIM = 354

# --- Feature block ranges (for ValHard occlusion) ---
FACE_SLICE = slice(0, 216)      # 72 * 3
POSE_SLICE = slice(216, 228)    # 4 * 3
LH_SLICE   = slice(228, 291)    # 21 * 3
RH_SLICE   = slice(291, 354)    # 21 * 3


def _flatten_frame_to_vec(frame: Dict[str, Any]) -> np.ndarray:
    """
    Convert one frame dict (from your extractor) into a flat [354] vector in a FIXED order:
      face (FACE_LANDMARKS order) → pose (POSE_LANDMARKS order) →
      left_hand (0..20) → right_hand (0..20), each as (x,y,z).
    Assumes your extractor already zero-fills when missing.
    """
    vec = []

    # Face (72 entries), in FACE_LANDMARKS order
    face_list = frame.get("face", [])
    face_map = {kp["id"]: (kp["x"], kp["y"], kp["z"]) for kp in face_list}
    for i in FACE_LANDMARKS:
        x, y, z = face_map.get(i, (0.0, 0.0, 0.0))
        vec.extend((x, y, z))

    # Pose (4 specific indices)
    pose_list = frame.get("pose", [])
    pose_map = {kp["id"]: (kp["x"], kp["y"], kp["z"]) for kp in pose_list}
    for i in POSE_LANDMARKS:
        x, y, z = pose_map.get(i, (0.0, 0.0, 0.0))
        vec.extend((x, y, z))

    # Left hand (0..20)
    lh_list = frame.get("left_hand", [])
    lh_map = {kp["id"]: (kp["x"], kp["y"], kp["z"]) for kp in lh_list}
    for i in range(HAND_LANDMARK_COUNT):
        x, y, z = lh_map.get(i, (0.0, 0.0, 0.0))
        vec.extend((x, y, z))

    # Right hand (0..20)
    rh_list = frame.get("right_hand", [])
    rh_map = {kp["id"]: (kp["x"], kp["y"], kp["z"]) for kp in rh_list}
    for i in range(HAND_LANDMARK_COUNT):
        x, y, z = rh_map.get(i, (0.0, 0.0, 0.0))
        vec.extend((x, y, z))

    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {arr.shape[0]}")
    return arr


def _video_to_TF(video_frames: List[Dict[str, Any]]) -> np.ndarray:
    """
    Convert a list of frame dicts (one video) into [T, 354].
    T is typically 20 (your extractor uses NUM_FRAMES=20), but we don't assume fixed T.
    """
    frames = [_flatten_frame_to_vec(f) for f in video_frames]  # list of [354]
    return np.stack(frames, axis=0).astype(np.float32)         # [T, 354]


class FixedSentenceJSONDataset(Dataset):
    """
    Build 10-word synthetic sentences directly from your JSON bundles:

      root_dir/
        hello.json             # { video_id: [frame_dict, ...], ... }
        my.json
        name.json
        ...

    - Only words listed in vocab_json (IDs 1..50) are used; 0 is reserved for CTC blank.
    - Each __getitem__ picks 10 words (with replacement), selects a random video for each,
      converts them to [T_i, 354], and concatenates along time → one long sentence.
    - No extra aug in train: your JSONs already contain Gaussian noise + horizontal translation.
    - Optional ValHard mode (validation realism): time-warp, boundary trim+crossfade, hand occlusions.
    """

    def __init__(
        self,
        root_dir: str,                   # e.g. "keypoints_aug_50/train" or ".../val"
        vocab_json: str,                 # mapping {word: id} with ids EXACTLY 1..50
        words_per_sentence: int = 10,    # fixed; assert enforced
        frame_gap: int = 0,              # optional zero frames between words
        seed: int = 42,
        load_into_memory: bool = True,   # True: pre-parse JSONs to RAM for speed
        valhard: bool = True            # NEW: enable realistic validation difficulty
    ):
        super().__init__()
        assert words_per_sentence == 10, "This dataset is fixed to 10-word sentences."
        self.root_dir = root_dir
        self.words_per_sentence = words_per_sentence
        self.frame_gap = frame_gap
        self.load_into_memory = load_into_memory
        self.valhard = valhard

        # ValHard parameters (used only if valhard=True)
        self.vh_time_warp_prob = 0.40
        self.vh_time_warp_scale = (0.8, 1.25)
        self.vh_occlusion_prob = 0.10  # 10% of words get a short LH/RH occlusion
        self.vh_crossfade_minmax = (1, 2)  # frames
        self.vh_trim_front_max = 2
        self.vh_trim_back_max = 3

        random.seed(seed)
        np.random.seed(seed)

        # --- Load & validate vocab (must be ids 1..50) ---
        if not (vocab_json and os.path.isfile(vocab_json)):
            raise FileNotFoundError("vocab_json is required and must exist.")
        with open(vocab_json, "r") as f:
            self.word2id: Dict[str, int] = {k: int(v) for k, v in json.load(f).items()}
        ids = sorted(set(self.word2id.values()))
        if ids != list(range(1, 51)):
            raise ValueError("vocab_json must contain IDs 1..50 exactly (0 is the CTC blank).")

        # --- Gather per-word JSONs ---
        json_paths = sorted(glob(os.path.join(root_dir, "*.json")))
        if not json_paths:
            raise FileNotFoundError(f"No JSON files found under {root_dir}")

        # Build index: word -> list of videos, where each video is [T, 354] (if loaded) or raw frames (lazy)
        self.words = sorted(self.word2id.keys())  # exactly 50 words expected
        self.index: Dict[str, List[Any]] = {w: [] for w in self.words}

        for jp in json_paths:
            word = os.path.splitext(os.path.basename(jp))[0]
            if word not in self.word2id:
                # Ignore non-top-50 words
                continue

            with open(jp, "r") as f:
                data = json.load(f)  # {video_id: [frame_dict, ...], ...}

            if not isinstance(data, dict):
                raise ValueError(f"{jp} must be a dict of video_id -> frames list.")

            for video_id, frames in data.items():
                if not isinstance(frames, list) or len(frames) == 0:
                    continue  # skip empty videos gracefully
                if self.load_into_memory:
                    arr = _video_to_TF(frames)        # [T, 354]
                    self.index[word].append(arr)
                else:
                    self.index[word].append(frames)

            if len(self.index[word]) == 0:
                raise ValueError(f"No usable videos parsed for word '{word}' in {jp}")

        # Check all 50 words exist (fail fast so experiments are reproducible)
        missing = [w for w in self.words if len(self.index[w]) == 0]
        if missing:
            raise FileNotFoundError(f"Missing or empty JSON for words: {missing}")

        # Virtual size → each call creates a new synthetic sentence
        self._size = 20000

    @property
    def vocab_size(self) -> int:
        return 50

    def __len__(self):
        return self._size

    # ---------- ValHard helpers ----------

    def _time_warp(self, clip: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self.vh_time_warp_prob:
            return clip
        T, F = clip.shape
        scale = np.random.uniform(*self.vh_time_warp_scale)
        new_T = max(1, int(round(T * scale)))
        idx = np.linspace(0, T - 1, num=new_T)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, T - 1)
        a = (idx - lo)[:, None]
        warped = (1.0 - a) * clip[lo] + a * clip[hi]
        return warped.astype(np.float32)

    def _trim_clip(self, clip: np.ndarray) -> np.ndarray:
        if clip.shape[0] <= 6:
            return clip
        trim_front = np.random.randint(0, self.vh_trim_front_max + 1)  # 0..2
        trim_back  = np.random.randint(0, self.vh_trim_back_max + 1)   # 0..3
        if clip.shape[0] - trim_front - trim_back >= 5:
            clip = clip[trim_front: clip.shape[0] - trim_back]
        return clip

    def _occlude_hand_burst(self, clip: np.ndarray) -> np.ndarray:
        if clip.shape[0] <= 8 or np.random.rand() >= self.vh_occlusion_prob:
            return clip
        T = clip.shape[0]
        start = np.random.randint(0, T - 5)
        length = np.random.randint(3, 7)  # 3..6
        if np.random.rand() < 0.5:
            clip[start:start+length, LH_SLICE] = 0.0
        else:
            clip[start:start+length, RH_SLICE] = 0.0
        return clip

    def _crossfade(self, prev: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if prev.shape[0] < 3 or curr.shape[0] < 3:
            return prev, curr
        k_min, k_max = self.vh_crossfade_minmax
        k = np.random.randint(k_min, k_max + 1)  # 1 or 2
        k = min(k, prev.shape[0], curr.shape[0])
        fade_in  = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        fade_out = 1.0 - fade_in
        blended = prev[-k:] * fade_out + curr[:k] * fade_in
        prev = np.concatenate([prev[:-k], blended], axis=0)
        curr = curr[k:]
        return prev, curr

    # ---------- Main sampling ----------

    def __getitem__(self, idx):
        # Pick exactly 10 words (with replacement so repeats are possible/realistic)
        chosen_words = random.choices(self.words, k=self.words_per_sentence)

        seqs: List[np.ndarray] = []
        labels: List[int] = []

        for i, w in enumerate(chosen_words):
            pool = self.index[w]
            src = random.choice(pool)

            # Resolve to [T, 354]
            if isinstance(src, np.ndarray):
                clip = src.copy()
            else:
                clip = _video_to_TF(src)

            # ValHard realism
            if self.valhard:
                # time-warp, trim, occlude
                clip = self._time_warp(clip)
                clip = self._trim_clip(clip)
                clip = self._occlude_hand_burst(clip)

            labels.append(self.word2id[w])

            # Crossfade with previous if ValHard (replaces explicit frame_gap)
            if self.valhard and len(seqs) > 0:
                prev = seqs[-1]
                prev, clip = self._crossfade(prev, clip)
                seqs[-1] = prev

            seqs.append(clip)

            # Optional explicit zero-gap frames (keep OFF for ValHard)
            if (not self.valhard) and self.frame_gap > 0 and i < self.words_per_sentence - 1:
                seqs.append(np.zeros((self.frame_gap, FEATURE_DIM), dtype=np.float32))

        sentence = np.concatenate(seqs, axis=0).astype(np.float32)  # [T_total, 354]
        labels = np.asarray(labels, dtype=np.int32)                  # [10]
        return torch.from_numpy(sentence), torch.from_numpy(labels)


def ctc_collate_fixed(batch):
    """
    Collate for CTC:
      inputs -> padded to [B, max_T, 354]
      input_lengths -> [B]
      targets -> concatenated 1D (length = 10 * B)
      target_lengths -> [B] (all = 10)
    """
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)  # should be all 10
    concat_targets = torch.cat([t for t in targets], dim=0).to(torch.long)
    return padded_inputs, input_lengths, concat_targets, target_lengths
