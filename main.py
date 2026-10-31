# ============================================================
# GYM AI TRACKER — FULL MAIN
# Features:
#   - AI Auto-Detect mode: biomechanics switch live per prediction
#   - Rep counted on UP (lockout/completion) position
#   - Deadlift: hip lockout + wrist height only (no knee/leg angles)
#   - Force/Power with per-exercise effective mass (mass_mode)
#   - Velocity loss, Fatigue index, Set RPE, Rep quality score
#   - Live force/velocity bar graph overlay
# ============================================================

import cv2
import mediapipe as mp
from angle_helper import calculate_angle

import time
import json
import math
import os
import csv
import joblib

from collections import deque

# ============================================================
# CONFIG
# ============================================================

VISIBILITY_THRESH    = 0.6
REP_COOLDOWN_SEC     = 0.8
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW     = 6
SET_REST_SECONDS     = 20
MIN_REPS_PER_SET     = 3
GRAVITY              = 9.81   # m/s²

DATASET_FILE    = "dataset.csv"
DATASET_HEADERS = [
    "knee_angle", "hip_angle", "elbow_angle", "shoulder_angle",
    "wrist_velocity", "hip_velocity", "label"
]

# ============================================================
# EFFECTIVE MASS CALCULATOR
#
# mass_mode options:
#   "full_bw_plus_load"       – squat, lunge: whole body + bar travel vertically
#   "load_only"               – bench, deadlift: body grounded, only bar moves
#   "upper_segment_plus_load" – curls, press: arm segment mass + implement
#                               arm segment (forearm+upper arm) ≈ 6.5 % BW
#                               forearm+hand only              ≈ 2.2 % BW
#
# References: Winter (2009) anthropometric segment data.
# ============================================================

# Fraction of BW that represents the moving arm segment for upper-body lifts.
# Shoulder press / lateral raise → full arm ≈ 6.5 % BW each side → 13 % total
# Bicep curl → forearm+hand ≈ 2.2 % BW each side → 4.4 % total
ARM_SEGMENT_FRACTIONS = {
    "bicep_curl":     0.044,   # both forearms+hands
    "shoulder_press": 0.130,   # both full arms
    "lateral_raise":  0.130,
    "custom":         0.0,     # unknown — use load_only as safe default
}

def compute_effective_mass(exercise_key: str, body_mass_kg: float,
                           external_load_kg: float) -> float:
    """
    Returns the effective moving mass (kg) to use in F = m(g+a).
    """
    cfg = EXERCISE_CONFIG.get(exercise_key, {})
    mode = cfg.get("mass_mode", "load_only")

    if mode == "full_bw_plus_load":
        return body_mass_kg + external_load_kg

    elif mode == "upper_segment_plus_load":
        frac = ARM_SEGMENT_FRACTIONS.get(exercise_key, 0.065)
        return body_mass_kg * frac + external_load_kg

    else:   # "load_only" — default / deadlift / bench
        # Guard: if no load entered, fall back to a minimum so force != 0
        return max(external_load_kg, 1.0)

# ============================================================
# EXERCISE CONFIG
# up/down = angle values. Rep counted when UP angle is reached.
# mass_mode controls which body segments count toward force/power.
# ============================================================

EXERCISE_CONFIG = {

    "squat": {
        "up": 165, "down": 115, "up_is_high": True,
        "label": "Squat", "rep_mode": "angle",
        # Entire system (BW + bar) moves vertically
        "mass_mode": "full_bw_plus_load",
    },
    "deadlift": {
        # shoulder→hip→knee angle; high=standing, low=hinge
        "up": 170, "down": 145, "up_is_high": True,
        "label": "Conventional Deadlift", "rep_mode": "deadlift",
        # Body is grounded — only the bar is lifted
        "mass_mode": "load_only",
    },
    "sumo_deadlift": {
        "up": 155, "down": 105, "up_is_high": True,
        "label": "Sumo Deadlift", "rep_mode": "deadlift",
        "mass_mode": "load_only",
    },
    "bench_press": {
        "up": 150, "down": 95, "up_is_high": True,
        "label": "Bench Press", "rep_mode": "angle",
        # Body supported by bench — only bar moves
        "mass_mode": "load_only",
    },
    "bicep_curl": {
        "up": 55, "down": 150, "up_is_high": False,
        "label": "Bicep Curl", "rep_mode": "angle",
        # Forearm+hand ≈ 2.2 % BW per arm + implement
        "mass_mode": "upper_segment_plus_load",
    },
    "shoulder_press": {
        "up": 170, "down": 80, "up_is_high": True,
        "label": "Shoulder Press", "rep_mode": "angle",
        # Full arm segment ≈ 6.5 % BW per arm + implement
        "mass_mode": "upper_segment_plus_load",
    },
    "lateral_raise": {
        "up": 150, "down": 30, "up_is_high": True,
        "label": "Lateral Raise", "rep_mode": "angle",
        "mass_mode": "upper_segment_plus_load",
    },
    "lunge": {
        "up": 160, "down": 100, "up_is_high": True,
        "label": "Lunge", "rep_mode": "angle",
        # Full BW + any external load
        "mass_mode": "full_bw_plus_load",
    },
    "custom": {
        "up": 90, "down": 150, "up_is_high": False,
        "label": "Custom Exercise", "rep_mode": "angle",
        "mass_mode": "load_only",
    },
}

# ============================================================
# EXERCISE LANDMARKS
# ============================================================

EXERCISE_LANDMARKS = {
    "squat":          ("LEFT_HIP",      "LEFT_KNEE",     "LEFT_ANKLE"),
    "deadlift":       ("LEFT_SHOULDER", "LEFT_HIP",      "LEFT_KNEE"),
    "sumo_deadlift":  ("LEFT_SHOULDER", "LEFT_HIP",      "LEFT_KNEE"),
    "bench_press":    ("LEFT_SHOULDER", "LEFT_ELBOW",    "LEFT_WRIST"),
    "bicep_curl":     ("LEFT_SHOULDER", "LEFT_ELBOW",    "LEFT_WRIST"),
    "shoulder_press": ("LEFT_SHOULDER", "LEFT_ELBOW",    "LEFT_WRIST"),
    "lateral_raise":  ("LEFT_HIP",      "LEFT_SHOULDER", "LEFT_ELBOW"),
    "lunge":          ("LEFT_HIP",      "LEFT_KNEE",     "LEFT_ANKLE"),
    "custom":         ("LEFT_HIP",      "LEFT_KNEE",     "LEFT_ANKLE"),
}

RIGHT_MIRROR = {
    "LEFT_HIP":      "RIGHT_HIP",
    "LEFT_KNEE":     "RIGHT_KNEE",
    "LEFT_ANKLE":    "RIGHT_ANKLE",
    "LEFT_SHOULDER": "RIGHT_SHOULDER",
    "LEFT_ELBOW":    "RIGHT_ELBOW",
    "LEFT_WRIST":    "RIGHT_WRIST",
}

ALL_LANDMARK_NAMES = [
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
]

# ============================================================
# RPE TABLE  (Gonzalez-Badillo & Sanchez-Medina)
# velocity loss %  →  approx RPE
# ============================================================

VL_RPE_TABLE = [
    (5,  6.0),
    (10, 7.0),
    (15, 7.5),
    (20, 8.0),
    (25, 8.5),
    (30, 9.0),
    (40, 9.5),
    (50, 10.0),
]

def velocity_loss_to_rpe(vl_pct: float) -> float:
    if vl_pct <= 0:
        return 5.0
    for i, (vl, rpe) in enumerate(VL_RPE_TABLE):
        if vl_pct <= vl:
            if i == 0:
                return rpe
            prev_vl, prev_rpe = VL_RPE_TABLE[i - 1]
            t = (vl_pct - prev_vl) / (vl - prev_vl)
            return round(prev_rpe + t * (rpe - prev_rpe), 1)
    return 10.0

# ============================================================
# STARTUP PROMPT
# ============================================================

def startup_prompt():

    print("\n" + "=" * 55)
    print("  GYM AI TRACKER — Setup")
    print("=" * 55)

    while True:
        try:
            body_mass = float(input("  Bodyweight (kg): ").strip())
            break
        except ValueError:
            print("  Enter a number.")

    while True:
        try:
            external_load = float(input("  Load / bar weight (kg, 0 if none): ").strip())
            break
        except ValueError:
            print("  Enter a number.")

    keys = list(EXERCISE_CONFIG.keys())

    print("\n  TRACKING MODES:")
    print("  " + "─" * 51)
    print("  0.  AI AUTO-DETECT  ← AI picks exercise & biomechanics")
    print("  " + "─" * 51)
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {EXERCISE_CONFIG[key]['label']}")
    print()

    while True:
        choice = input("Pick mode/exercise number: ").strip()
        if choice == "0":
            return _setup_ai_mode(body_mass, external_load)
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                exercise_key = keys[idx]
                break

    cfg       = EXERCISE_CONFIG[exercise_key].copy()
    landmarks = EXERCISE_LANDMARKS[exercise_key]

    if exercise_key == "custom":
        print("\n── Custom Exercise ──────────────────────────────")
        cfg["label"] = input("Exercise name: ").strip() or "Custom"
        print("\nPick 3 landmarks for the joint angle:")
        for i, n in enumerate(ALL_LANDMARK_NAMES, 1):
            print(f"  {i}. {n}")

        def pick_lm(prompt):
            while True:
                r = input(prompt).strip()
                if r.isdigit() and 1 <= int(r) <= len(ALL_LANDMARK_NAMES):
                    return ALL_LANDMARK_NAMES[int(r) - 1]
                print("  Enter a number from the list.")

        a = pick_lm("  Point A (e.g. hip): ")
        b = pick_lm("  Point B — the joint (e.g. knee): ")
        c = pick_lm("  Point C (e.g. ankle): ")
        landmarks = (a, b, c)

        cfg["up"]         = float(input("Angle for UP position: "))
        cfg["down"]       = float(input("Angle for DOWN position: "))
        hi                = input("Is UP the HIGH angle? (y/n): ").strip().lower()
        cfg["up_is_high"] = (hi == "y")
        cfg["rep_mode"]   = "angle"
        cfg["mass_mode"]  = "load_only"

    exercise_label = input(
        f"\nLabel for your exercise (default '{exercise_key}'): "
    ).strip() or exercise_key
    rest_label = "rest"

    eff_mass = compute_effective_mass(exercise_key, body_mass, external_load)
    print(f"\n✅  Ready! — Manual mode: {cfg['label']}")
    print(f"   Effective mass for force/power: {eff_mass:.1f} kg")
    _print_keys(exercise_label, rest_label)

    return (exercise_key, cfg, landmarks, exercise_label, rest_label,
            False, body_mass, external_load)


def _setup_ai_mode(body_mass, external_load):
    exercise_label = input(
        "\nLabel for exercise data collection (default 'exercise'): "
    ).strip() or "exercise"
    rest_label = "rest"

    cfg       = EXERCISE_CONFIG["squat"].copy()
    landmarks = EXERCISE_LANDMARKS["squat"]

    print(f"\n✅  Ready! — AI AUTO-DETECT mode")
    print("   Biomechanics + effective mass will switch with each AI prediction.")
    _print_keys(exercise_label, rest_label)

    return ("ai_auto", cfg, landmarks, exercise_label, rest_label,
            True, body_mass, external_load)


def _print_keys(exercise_label, rest_label):
    print(f"\n   Key 1 : hold while exercising  → saves '{exercise_label}'")
    print(f"   Key 2 : hold while at rest      → saves '{rest_label}'")
    print(f"   Q     : quit and save session")
    print("=" * 55 + "\n")

# ============================================================
# DATASET CSV
# ============================================================

def init_csv():
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            csv.writer(f).writerow(DATASET_HEADERS)
        print(f"[dataset] Created {DATASET_FILE}")
    else:
        print(f"[dataset] Appending to existing {DATASET_FILE}")

def save_example(features: dict, label: str):
    row = [
        round(features.get("knee_angle",     0), 2),
        round(features.get("hip_angle",      0), 2),
        round(features.get("elbow_angle",    0), 2),
        round(features.get("shoulder_angle", 0), 2),
        round(features.get("wrist_velocity", 0), 2),
        round(features.get("hip_velocity",   0), 2),
        label,
    ]
    with open(DATASET_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ============================================================
# KALMAN FILTER
# ============================================================

class KalmanAngleFilter:

    def __init__(self, process_noise=1.0, measurement_noise=8.0):
        self.x       = None
        self.P       = [[100, 0], [0, 100]]
        self.q_angle = process_noise
        self.q_vel   = process_noise * 0.1
        self.R       = measurement_noise
        self.last_t  = None

    def update(self, measurement):
        now = time.perf_counter()
        if self.x is None:
            self.x      = [measurement, 0.0]
            self.last_t = now
            return measurement, 0.0

        dt = max(now - self.last_t, 1e-3)
        self.last_t = now

        ap  = self.x[0] + self.x[1] * dt
        vp  = self.x[1]

        p00 = (self.P[0][0] + dt * (self.P[1][0] + self.P[0][1])
               + dt**2 * self.P[1][1] + self.q_angle)
        p01 = self.P[0][1] + dt * self.P[1][1]
        p10 = self.P[1][0] + dt * self.P[1][1]
        p11 = self.P[1][1] + self.q_vel

        S   = p00 + self.R
        K0  = p00 / S
        K1  = p10 / S
        inn = measurement - ap

        self.x = [ap + K0 * inn, vp + K1 * inn]
        self.P = [
            [(1 - K0) * p00, (1 - K0) * p01],
            [p10 - K1 * p00,  p11 - K1 * p01],
        ]
        return self.x[0], self.x[1]

# ============================================================
# PER-REP METRICS
# ============================================================

class RepMetrics:

    def __init__(self):
        self.reset()

    def reset(self):
        self._samples = []

    def record(self, angle, velocity):
        self._samples.append((time.perf_counter(), angle, velocity))

    def compute(self, phase_split_angle, up_is_high):
        if len(self._samples) < 2:
            return {}

        times  = [s[0] for s in self._samples]
        angles = [s[1] for s in self._samples]
        vels   = [s[2] for s in self._samples]

        duration = times[-1] - times[0]
        rom      = max(angles) - min(angles)
        si       = angles.index(min(angles))

        ct = times[:si + 1] if si > 0 else times
        et = times[si:]     if si < len(times) - 1 else times
        ca = angles[:si + 1] if si > 0 else angles
        ea = angles[si:]     if si < len(angles) - 1 else angles

        cd = (ct[-1] - ct[0]) if len(ct) > 1 else 0
        ed = (et[-1] - et[0]) if len(et) > 1 else 0
        cs = abs(ca[-1] - ca[0]) / cd if cd > 0 else 0
        es = abs(ea[-1] - ea[0]) / ed if ed > 0 else 0
        pv = max(abs(v) for v in vels) if vels else 0

        return {
            "duration_s":     round(duration, 2),
            "rom_deg":        round(rom, 1),
            "tempo":          f"{ed:.1f}-0.0-{cd:.1f}",
            "conc_dur_s":     round(cd, 2),
            "ecc_dur_s":      round(ed, 2),
            "conc_speed_dps": round(cs, 1),
            "ecc_speed_dps":  round(es, 1),
            "peak_vel_dps":   round(pv, 1),
        }

# ============================================================
# VBT ENGINE
# ============================================================

class VBTEngine:
    """
    Velocity-Based Training metrics.

    Velocity (deg/s) → linear proxy via DEG_S_TO_M_S scale factor.
    1 deg/s ≈ 0.003 m/s — conservative mid-ROM estimate for compound lifts.
    Effective mass is set per exercise via set_mass().
    """

    DEG_S_TO_M_S = 0.003

    def __init__(self, effective_mass_kg: float):
        self.mass          = effective_mass_kg
        self.first_rep_vel = None
        self.rep_peak_vels = []
        self.rep_forces    = []
        self.rep_powers    = []
        self.rep_vl_pcts   = []
        self.fatigue_index = 0.0
        self.set_rpe       = 5.0

    def set_mass(self, effective_mass_kg: float):
        """Update effective mass (called when AI switches exercise)."""
        self.mass = effective_mass_kg

    def record_rep(self, peak_vel_dps: float, peak_acc_dps2: float) -> dict:
        v_ms  = abs(peak_vel_dps)  * self.DEG_S_TO_M_S
        a_ms2 = abs(peak_acc_dps2) * self.DEG_S_TO_M_S

        force = self.mass * (GRAVITY + a_ms2)
        power = force * v_ms

        if self.first_rep_vel is None:
            self.first_rep_vel = abs(peak_vel_dps) if peak_vel_dps != 0 else 1.0

        vl_pct = max(0.0,
            (self.first_rep_vel - abs(peak_vel_dps)) / self.first_rep_vel * 100.0)

        self.rep_peak_vels.append(abs(peak_vel_dps))
        self.rep_forces.append(force)
        self.rep_powers.append(power)
        self.rep_vl_pcts.append(vl_pct)

        if len(self.rep_peak_vels) > 1:
            self.fatigue_index = max(0.0,
                (self.rep_peak_vels[0] - self.rep_peak_vels[-1])
                / self.rep_peak_vels[0] * 100.0)

        self.set_rpe = velocity_loss_to_rpe(vl_pct)

        return {
            "peak_vel_dps": round(abs(peak_vel_dps), 1),
            "peak_vel_ms":  round(v_ms, 3),
            "force_N":      round(force, 1),
            "power_W":      round(power, 1),
            "vl_pct":       round(vl_pct, 1),
            "fatigue_idx":  round(self.fatigue_index, 1),
            "set_rpe":      self.set_rpe,
        }

    def reset_set(self):
        self.first_rep_vel = None
        self.rep_peak_vels.clear()
        self.rep_forces.clear()
        self.rep_powers.clear()
        self.rep_vl_pcts.clear()
        self.fatigue_index = 0.0
        self.set_rpe       = 5.0

    @property
    def last_force(self):
        return self.rep_forces[-1] if self.rep_forces else 0.0

    @property
    def last_power(self):
        return self.rep_powers[-1] if self.rep_powers else 0.0

    @property
    def last_vl_pct(self):
        return self.rep_vl_pcts[-1] if self.rep_vl_pcts else 0.0

# ============================================================
# REP QUALITY SCORER
# ============================================================

def score_rep_quality(
    rom_deg: float,
    peak_vel_dps: float,
    vl_pct: float,
    left_counter: int,
    right_counter: int,
    target_rom: float = 50.0,
) -> tuple:
    """
    Returns (score 0-100, label string).

    40 pts — ROM vs target
    30 pts — Peak velocity (cap at 400 dps)
    20 pts — Velocity loss penalty
    10 pts — Left/right symmetry
    """
    rom_score = min(rom_deg / target_rom, 1.0) * 40.0
    vel_score = min(abs(peak_vel_dps) / 400.0, 1.0) * 30.0
    vl_score  = max(0.0, 1.0 - vl_pct / 50.0) * 20.0
    sym_diff  = abs(left_counter - right_counter)
    sym_score = max(0.0, 1.0 - sym_diff / 3.0) * 10.0

    total = int(rom_score + vel_score + vl_score + sym_score)

    if total >= 85:   label = "Excellent"
    elif total >= 70: label = "Good"
    elif total >= 50: label = "Fair"
    else:             label = "Poor"

    return total, label

# ============================================================
# REP TRACKER
# ============================================================

class SideTracker:

    WRIST_RISE_THRESH = 0.10   # ≥10 % of frame height for deadlift

    def __init__(self, config, side="left"):
        self.config       = config
        self.side         = side
        self.stage        = "down"
        self.counter      = 0
        self.last_rep_t   = 0.0
        self.armed        = False

        self.angle_buf    = deque(maxlen=SMOOTHING_WINDOW)
        self.smooth_angle = 0.0
        self.kalman       = KalmanAngleFilter(1.0, 8.0)
        self.kalman_angle = 0.0
        self.velocity     = 0.0

        self._prev_vel    = 0.0
        self._prev_vel_t  = None
        self.peak_acc     = 0.0

        self.wrist_y_low  = None
        self.wrist_y_now  = None

        self.rep_metrics  = RepMetrics()
        self.rep_history  = []
        self.last_vbt     = {}

    def reconfigure(self, new_config):
        if new_config is self.config:
            return
        self.config      = new_config
        self.stage       = "down"
        self.armed       = False
        self.wrist_y_low = None
        self.peak_acc    = 0.0

    def update(self, raw_angle, wrist_y=None):
        self.angle_buf.append(raw_angle)
        self.smooth_angle = sum(self.angle_buf) / len(self.angle_buf)
        self.kalman_angle, self.velocity = self.kalman.update(raw_angle)
        self.rep_metrics.record(self.kalman_angle, self.velocity)

        now = time.perf_counter()
        if self._prev_vel_t is not None:
            dt  = max(now - self._prev_vel_t, 1e-3)
            acc = (self.velocity - self._prev_vel) / dt
            if abs(acc) > abs(self.peak_acc):
                self.peak_acc = acc
        self._prev_vel   = self.velocity
        self._prev_vel_t = now

        self.wrist_y_now = wrist_y

        cfg = self.config
        if cfg.get("rep_mode") == "deadlift":
            return self._update_deadlift(cfg, now)
        return self._update_angle(cfg, now)

    # ── Standard angle-based rep ──────────────────────────────

    def _update_angle(self, cfg, now):
        angle = self.kalman_angle

        if cfg["up_is_high"]:
            if angle <= cfg["down"]:
                self.armed = True
                self.stage = "down"
            if angle >= cfg["up"] and self.armed:
                self.stage = "up"
                if now - self.last_rep_t > REP_COOLDOWN_SEC:
                    return self._finish_rep(now, cfg)
        else:
            if angle >= cfg["down"]:
                self.armed = True
                self.stage = "down"
            if angle <= cfg["up"] and self.armed:
                self.stage = "up"
                if now - self.last_rep_t > REP_COOLDOWN_SEC:
                    return self._finish_rep(now, cfg)

        return False

    # ── Deadlift: hip lockout + wrist travelled upward ────────
    #
    # State machine:
    #   IDLE → DOWN : hip angle falls below cfg["down"]  (hinge detected)
    #   DOWN → UP   : hip angle rises above cfg["up"]    (lockout)
    #                 AND wrist has risen ≥ WRIST_RISE_THRESH
    #   UP   → DOWN : hip drops again, ready for next rep
    # ──────────────────────────────────────────────────────────

    def _update_deadlift(self, cfg, now):
        hip_angle = self.kalman_angle
        wrist_y   = self.wrist_y_now

        # Phase 1: arm on hinge
        if self.stage != "up":
            if hip_angle <= cfg["down"]:
                self.stage = "down"
                self.armed = True
                if wrist_y is not None:
                    if self.wrist_y_low is None or wrist_y > self.wrist_y_low:
                        self.wrist_y_low = wrist_y

        # Phase 2: lockout check
        if self.armed and self.stage == "down" and hip_angle >= cfg["up"]:
            rise = (
                (self.wrist_y_low - wrist_y)
                if (wrist_y is not None and self.wrist_y_low is not None)
                else 0.0
            )
            if rise >= self.WRIST_RISE_THRESH:
                self.stage = "up"
                if now - self.last_rep_t > REP_COOLDOWN_SEC:
                    saved_low        = self.wrist_y_low
                    self.wrist_y_low = None
                    self.armed       = False
                    counted = self._finish_rep(now, cfg)
                    if counted:
                        print(
                            f"      [deadlift] hip={hip_angle:.1f}  "
                            f"wrist_low={saved_low:.3f}  "
                            f"wrist_now={wrist_y:.3f}  rise={rise:.3f}"
                        )
                    return counted

        # Reset for next rep
        if self.stage == "up" and hip_angle <= cfg["down"]:
            self.stage       = "down"
            self.armed       = True
            self.wrist_y_low = wrist_y

        return False

    # ── Shared finish ─────────────────────────────────────────

    def _finish_rep(self, now, cfg):
        self.armed      = False
        self.stage      = "down"
        self.counter   += 1
        self.last_rep_t = now

        m             = self.rep_metrics.compute(
            (cfg["up"] + cfg["down"]) / 2, cfg["up_is_high"]
        )
        m["rep"]      = self.counter
        m["peak_acc"] = round(self.peak_acc, 1)
        self.rep_history.append(m)
        self.rep_metrics.reset()
        self.peak_acc = 0.0

        print(
            f"    [{self.side.upper()}] Rep #{self.counter} ✓  "
            f"ROM={m.get('rom_deg')}  dur={m.get('duration_s')}s  "
            f"tempo={m.get('tempo')}  conc={m.get('conc_speed_dps')} dps"
        )
        return True

# ============================================================
# WORKOUT SESSION
# ============================================================

class WorkoutSession:

    def __init__(self):
        self.current_set      = 1
        self.current_set_reps = 0
        self.completed_sets   = []
        self.last_rep_time    = None
        self.rest_countdown   = SET_REST_SECONDS

    def register_rep(self):
        self.current_set_reps += 1
        self.last_rep_time     = time.time()

    def update(self):
        if self.last_rep_time is None:
            return
        elapsed             = time.time() - self.last_rep_time
        self.rest_countdown = max(0, int(SET_REST_SECONDS - elapsed))
        if elapsed >= SET_REST_SECONDS and self.current_set_reps >= MIN_REPS_PER_SET:
            self.finish_set()

    def finish_set(self):
        self.completed_sets.append({
            "set":       self.current_set,
            "reps":      self.current_set_reps,
            "timestamp": time.strftime("%H:%M:%S"),
        })
        print(f"\n✅ Set {self.current_set} completed ({self.current_set_reps} reps)")
        self.current_set      += 1
        self.current_set_reps  = 0
        self.last_rep_time     = None
        self.rest_countdown    = SET_REST_SECONDS

# ============================================================
# LANDMARK HELPERS
# ============================================================

def get_point(landmarks, name):
    lm = landmarks[getattr(mp_pose.PoseLandmark, name).value]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return [lm.x, lm.y]

def get_angle_for_side(landmarks, side, exercise_landmarks):
    a, b, c = exercise_landmarks
    if side == "right":
        a = RIGHT_MIRROR.get(a, a)
        b = RIGHT_MIRROR.get(b, b)
        c = RIGHT_MIRROR.get(c, c)
    pa = get_point(landmarks, a)
    pb = get_point(landmarks, b)
    pc = get_point(landmarks, c)
    if None in (pa, pb, pc):
        return None
    return calculate_angle(pa, pb, pc)

def get_wrist_y(landmarks, side):
    name = "LEFT_WRIST" if side == "left" else "RIGHT_WRIST"
    pt   = get_point(landmarks, name)
    return pt[1] if pt else None

# ============================================================
# FEATURE EXTRACTION
# ============================================================

_wrist_kalman = KalmanAngleFilter(2.0, 5.0)
_hip_kalman   = KalmanAngleFilter(2.0, 5.0)
_prev_wrist   = None
_prev_hip     = None
_prev_feat_t  = None

def extract_features(landmarks):
    global _prev_wrist, _prev_hip, _prev_feat_t

    def af(a, b, c):
        pa = get_point(landmarks, a)
        pb = get_point(landmarks, b)
        pc = get_point(landmarks, c)
        return calculate_angle(pa, pb, pc) if None not in (pa, pb, pc) else 0.0

    knee_angle     = af("LEFT_HIP",      "LEFT_KNEE",     "LEFT_ANKLE")
    hip_angle      = af("LEFT_SHOULDER", "LEFT_HIP",      "LEFT_KNEE")
    elbow_angle    = af("LEFT_SHOULDER", "LEFT_ELBOW",    "LEFT_WRIST")
    shoulder_angle = af("LEFT_HIP",      "LEFT_SHOULDER", "LEFT_ELBOW")

    now   = time.perf_counter()
    wrist = get_point(landmarks, "LEFT_WRIST")
    hip   = get_point(landmarks, "LEFT_HIP")
    wv = hv = 0.0

    if _prev_feat_t is not None:
        dt = max(now - _prev_feat_t, 1e-3)
        if wrist and _prev_wrist:
            raw = math.hypot(
                (wrist[0] - _prev_wrist[0]) * 1000,
                (wrist[1] - _prev_wrist[1]) * 1000,
            ) / dt
            _, wv = _wrist_kalman.update(raw)
            wv = abs(wv)
        if hip and _prev_hip:
            raw = math.hypot(
                (hip[0] - _prev_hip[0]) * 1000,
                (hip[1] - _prev_hip[1]) * 1000,
            ) / dt
            _, hv = _hip_kalman.update(raw)
            hv = abs(hv)

    _prev_wrist  = wrist
    _prev_hip    = hip
    _prev_feat_t = now

    return {
        "knee_angle":     round(knee_angle,     1),
        "hip_angle":      round(hip_angle,      1),
        "elbow_angle":    round(elbow_angle,    1),
        "shoulder_angle": round(shoulder_angle, 1),
        "wrist_velocity": round(wv,             2),
        "hip_velocity":   round(hv,             2),
    }

# ============================================================
# LIVE FORCE / VELOCITY BAR GRAPH
# ============================================================

def draw_vbt_graph(frame, vbt, x0, y0,
                   bar_w=18, max_bars=10,
                   max_force=2000.0, max_power=1000.0):
    graph_h = 95
    graph_w = max_bars * (bar_w + 4) + 10

    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + graph_w, y0 + graph_h + 32), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "F(N) / P(W) per rep",
                (x0 + 2, y0 + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    n = min(len(vbt.rep_forces), max_bars)
    if n == 0:
        cv2.putText(frame, "no reps yet",
                    (x0 + 6, y0 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
        return

    forces  = vbt.rep_forces[-n:]
    powers  = vbt.rep_powers[-n:]
    vl_pcts = vbt.rep_vl_pcts[-n:]

    bar_top = y0 + 18
    bar_bot = y0 + graph_h + 12
    vl_pts  = []

    for i, (f, p, vl) in enumerate(zip(forces, powers, vl_pcts)):
        bx = x0 + 3 + i * (bar_w + 4)

        fh = int(min(f / max_force, 1.0) * (bar_bot - bar_top))
        cv2.rectangle(frame, (bx, bar_bot - fh),
                      (bx + bar_w // 2 - 1, bar_bot), (200, 100, 50), -1)

        ph = int(min(p / max_power, 1.0) * (bar_bot - bar_top))
        cv2.rectangle(frame, (bx + bar_w // 2, bar_bot - ph),
                      (bx + bar_w - 1, bar_bot), (60, 200, 80), -1)

        vl_y = bar_bot - int(min(vl / 50.0, 1.0) * (bar_bot - bar_top))
        vl_pts.append((bx + bar_w // 2, vl_y))

    if len(vl_pts) > 1:
        for i in range(len(vl_pts) - 1):
            cv2.line(frame, vl_pts[i], vl_pts[i + 1], (0, 60, 255), 2)
    for pt in vl_pts:
        cv2.circle(frame, pt, 3, (0, 100, 255), -1)

    lx = x0 + 2
    ly = y0 + graph_h + 28
    cv2.rectangle(frame, (lx, ly - 7), (lx + 8, ly), (200, 100, 50), -1)
    cv2.putText(frame, "F", (lx + 10, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 100, 50), 1)
    cv2.rectangle(frame, (lx + 20, ly - 7), (lx + 28, ly), (60, 200, 80), -1)
    cv2.putText(frame, "P", (lx + 30, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (60, 200, 80), 1)
    cv2.circle(frame, (lx + 43, ly - 4), 3, (0, 100, 255), -1)
    cv2.putText(frame, "VL%", (lx + 48, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 100, 255), 1)

# ============================================================
# OVERLAY
# ============================================================

# Right panel anchor — pulled left so nothing clips off-screen.
# All right-panel elements are drawn relative to RP_X.
RP_X       = 20          # pixels from the RIGHT edge to start the panel
RP_WIDTH   = 230         # panel content width in pixels

def draw_overlay(
    frame, left, right,
    predicted_exercise, confidence,
    cfg, lost_pose, session, features,
    collecting_label, collected_count,
    exercise_label, rest_label,
    ai_mode, vbt, last_rep_quality,
):
    h, w = frame.shape[:2]

    # Right panel X is computed from the right edge so it works at any resolution
    rx = w - RP_WIDTH - RP_X

    # ── Top banner ───────────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 215), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    total    = left.counter + right.counter
    mode_tag = "  [AI AUTO]" if ai_mode else ""

    cv2.putText(frame, cfg["label"] + mode_tag,
                (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 220, 60), 2)
    cv2.putText(frame, f"AI Detected: {predicted_exercise}  ({confidence:.0%})",
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    if ai_mode:
        cv2.putText(frame, "Biomechanics: following AI prediction",
                    (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 100, 255), 1)

    cv2.putText(frame, f"Reps  {total}",
                (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)

    # Per-side tracker rows
    for row_y, tracker, side in ((150, left, "L"), (172, right, "R")):
        ang  = f"{int(tracker.kalman_angle)}" if tracker.angle_buf else "--"
        stg  = tracker.stage or "--"
        extra = ""
        if cfg.get("rep_mode") == "deadlift":
            wy   = tracker.wrist_y_now
            wlow = tracker.wrist_y_low
            rise = round(wlow - wy, 2) if (wy is not None and wlow is not None) else 0.0
            extra = f"  rise:{rise:.2f} arm:{int(tracker.armed)}"
        cv2.putText(
            frame,
            f"{side}  {ang}°  [{stg}]  x{tracker.counter}"
            f"  vel:{tracker.velocity:+.0f}dps{extra}",
            (20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1,
        )

    # Last rep summary
    last_m = None
    for t in (left, right):
        if t.rep_history:
            m = t.rep_history[-1]
            if last_m is None or m["rep"] > last_m.get("rep", 0):
                last_m = m
    if last_m:
        cv2.putText(
            frame,
            f"ROM={last_m.get('rom_deg')}°  "
            f"dur={last_m.get('duration_s')}s  "
            f"tempo={last_m.get('tempo')}",
            (20, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 190, 80), 1,
        )

    # ── Right panel ───────────────────────────────────────────
    # Background strip
    rp_ov = frame.copy()
    cv2.rectangle(rp_ov, (rx - 8, 0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(rp_ov, 0.50, frame, 0.50, 0, frame)

    # Set / rest info
    cv2.putText(frame, f"Set: {session.current_set}",
                (rx, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 200, 100), 2)
    cv2.putText(frame, f"Set Reps: {session.current_set_reps}",
                (rx, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(frame, f"Rest: {session.rest_countdown}s",
                (rx, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 255), 1)

    # VBT section header
    vbt_y = 108
    cv2.putText(frame, "VBT METRICS",
                (rx, vbt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1)

    cv2.putText(frame, f"Force : {vbt.last_force:.0f} N",
                (rx, vbt_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 120, 60), 2)
    cv2.putText(frame, f"Power : {vbt.last_power:.0f} W",
                (rx, vbt_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 100), 2)

    vl = vbt.last_vl_pct
    vl_col = (0, 200, 80) if vl < 15 else (0, 180, 255) if vl < 30 else (0, 60, 255)
    cv2.putText(frame, f"Vel Loss: {vl:.1f}%",
                (rx, vbt_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, vl_col, 2)

    fi = vbt.fatigue_index
    fi_col = (0, 200, 80) if fi < 10 else (0, 180, 255) if fi < 25 else (0, 60, 255)
    cv2.putText(frame, f"Fatigue : {fi:.1f}%",
                (rx, vbt_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, fi_col, 2)

    rpe = vbt.set_rpe
    rpe_col = (0, 200, 80) if rpe < 7 else (0, 180, 255) if rpe < 8.5 else (0, 60, 255)
    cv2.putText(frame, f"Est RPE: {rpe:.1f}",
                (rx, vbt_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.68, rpe_col, 2)

    if last_rep_quality:
        q_score, q_label = last_rep_quality
        q_col = (0, 200, 80) if q_score >= 70 else (0, 180, 255) if q_score >= 50 else (0, 60, 255)
        cv2.putText(frame, f"Quality: {q_score}/100",
                    (rx, vbt_y + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.58, q_col, 2)
        cv2.putText(frame, f"  [{q_label}]",
                    (rx, vbt_y + 164), cv2.FONT_HERSHEY_SIMPLEX, 0.52, q_col, 1)

    # Bar graph — sits at the bottom of the right panel
    graph_y = h - 145
    draw_vbt_graph(frame, vbt, rx, graph_y)

    # ── Pose-lost warning ─────────────────────────────────────
    if lost_pose:
        cv2.putText(frame, "! Pose not detected",
                    (w // 2 - 150, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

    # ── Bottom-left: ML features ──────────────────────────────
    if features:
        py  = h - 120
        feat_w = rx - 16
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, py - 15), (feat_w, h), (10, 10, 10), -1)
        cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "ML FEATURES",
                    (10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(
            frame,
            f"knee:{features.get('knee_angle', 0):.0f}  "
            f"hip:{features.get('hip_angle', 0):.0f}  "
            f"elbow:{features.get('elbow_angle', 0):.0f}  "
            f"shoulder:{features.get('shoulder_angle', 0):.0f}",
            (10, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 255, 160), 1,
        )
        cv2.putText(
            frame,
            f"wrist_vel:{features.get('wrist_velocity', 0):.1f}  "
            f"hip_vel:{features.get('hip_velocity', 0):.1f}",
            (10, py + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 255), 1,
        )
        cv2.putText(
            frame,
            f"HOLD  1={exercise_label}  |  2={rest_label}",
            (10, py + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 100), 1,
        )

    # ── Recording border ──────────────────────────────────────
    if collecting_label:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 180), 4)
        cv2.putText(
            frame,
            f"REC  [{collecting_label}]  saved: {collected_count}",
            (w // 2 - 160, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 60, 255), 2,
        )

# ============================================================
# MAIN
# ============================================================

def main():

    result = startup_prompt()
    (exercise_key, cfg, exercise_landmarks,
     exercise_label, rest_label,
     ai_mode, body_mass, external_load) = result

    init_csv()

    if not os.path.exists("exercise_model.pkl"):
        print("exercise_model.pkl not found — run train.py first.")
        return

    model = joblib.load("exercise_model.pkl")
    print("[ML] Model loaded")

    # Compute initial effective mass for the chosen exercise
    eff_mass = compute_effective_mass(exercise_key, body_mass, external_load)

    left    = SideTracker(cfg, side="left")
    right   = SideTracker(cfg, side="right")
    session = WorkoutSession()
    vbt     = VBTEngine(eff_mass)

    key_to_label = {ord("1"): exercise_label, ord("2"): rest_label}

    prediction_buffer     = deque(maxlen=20)
    predicted_exercise    = "..."
    prediction_confidence = 0.0
    current_ai_key        = None

    last_rep_quality = None

    mp_pose_module = mp.solutions.pose
    mp_drawing     = mp.solutions.drawing_utils

    global mp_pose
    mp_pose = mp_pose_module

    pose = mp_pose_module.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    mode_str = "AI AUTO-DETECT" if ai_mode else cfg["label"]
    print(f"Tracking: {mode_str} | Hold 1={exercise_label}, 2=rest | Q=quit\n")

    lost_pose_frames = 0
    LOST_POSE_LIMIT  = 10
    current_features = {}
    collecting_label = None
    collected_count  = 0

    while True:

        ok, frame = cap.read()
        if not ok:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:

            lost_pose_frames = 0
            lm               = results.pose_landmarks.landmark

            # Feature extraction
            current_features = extract_features(lm)
            feat_list = [
                current_features["knee_angle"],
                current_features["hip_angle"],
                current_features["elbow_angle"],
                current_features["shoulder_angle"],
                current_features["wrist_velocity"],
                current_features["hip_velocity"],
            ]

            # ML prediction
            probabilities         = model.predict_proba([feat_list])[0]
            prediction_confidence = float(max(probabilities))
            prediction            = model.classes_[probabilities.argmax()]

            if prediction_confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(prediction)
                predicted_exercise = max(
                    set(prediction_buffer), key=prediction_buffer.count
                )

            # AI AUTO mode: swap biomechanics + effective mass live
            if ai_mode and predicted_exercise in EXERCISE_CONFIG:
                stable_pred = predicted_exercise
                if stable_pred != current_ai_key:
                    current_ai_key     = stable_pred
                    new_cfg            = EXERCISE_CONFIG[stable_pred].copy()
                    exercise_landmarks = EXERCISE_LANDMARKS[stable_pred]
                    cfg                = new_cfg
                    left.reconfigure(new_cfg)
                    right.reconfigure(new_cfg)

                    # Update effective mass for the new exercise
                    new_mass = compute_effective_mass(
                        stable_pred, body_mass, external_load
                    )
                    vbt.set_mass(new_mass)
                    vbt.reset_set()   # new baseline for velocity loss

                    print(
                        f"[AI] Switched → {new_cfg['label']}  "
                        f"eff_mass={new_mass:.1f} kg"
                    )

            # Rep counting
            for side, tracker in (("left", left), ("right", right)):
                angle   = get_angle_for_side(lm, side, exercise_landmarks)
                wrist_y = get_wrist_y(lm, side)
                if angle is not None:
                    if tracker.update(angle, wrist_y=wrist_y):
                        session.register_rep()

                        rep_hist = tracker.rep_history[-1]
                        peak_vel = rep_hist.get("peak_vel_dps", 0.0)
                        peak_acc = rep_hist.get("peak_acc",     0.0)
                        vbt_m    = vbt.record_rep(peak_vel, peak_acc)
                        tracker.last_vbt = vbt_m

                        q_score, q_label = score_rep_quality(
                            rom_deg       = rep_hist.get("rom_deg", 0.0),
                            peak_vel_dps  = peak_vel,
                            vl_pct        = vbt_m["vl_pct"],
                            left_counter  = left.counter,
                            right_counter = right.counter,
                        )
                        last_rep_quality = (q_score, q_label)

                        print(
                            f"      VBT  F={vbt_m['force_N']:.0f}N  "
                            f"P={vbt_m['power_W']:.0f}W  "
                            f"VL={vbt_m['vl_pct']:.1f}%  "
                            f"Fatigue={vbt_m['fatigue_idx']:.1f}%  "
                            f"RPE≈{vbt_m['set_rpe']}  "
                            f"Quality={q_score}/100 [{q_label}]"
                        )

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose_module.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245, 66, 230),  thickness=2),
            )

        else:
            lost_pose_frames += 1

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in key_to_label:
            collecting_label = key_to_label[key]
            if current_features:
                save_example(current_features, collecting_label)
                collected_count += 1
                if collected_count % 10 == 0:
                    print(f"  [dataset] {collected_count} examples  label={collecting_label}")
        else:
            collecting_label = None

        draw_overlay(
            frame, left, right,
            predicted_exercise, prediction_confidence,
            cfg, lost_pose_frames >= LOST_POSE_LIMIT,
            session, current_features,
            collecting_label, collected_count,
            exercise_label, rest_label,
            ai_mode, vbt, last_rep_quality,
        )

        cv2.imshow("Gym AI Tracker", frame)
        session.update()

    # Save session JSON
    output = {
        "exercise":       exercise_key,
        "label":          cfg["label"],
        "ai_mode":        ai_mode,
        "body_mass_kg":   body_mass,
        "external_load_kg": external_load,
        "eff_mass_kg":    vbt.mass,
        "sets":           session.completed_sets,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rep_metrics": {
            "left":  left.rep_history,
            "right": right.rep_history,
        },
        "vbt_summary": {
            "rep_forces_N":  [round(f, 1) for f in vbt.rep_forces],
            "rep_powers_W":  [round(p, 1) for p in vbt.rep_powers],
            "rep_vl_pct":    [round(v, 1) for v in vbt.rep_vl_pcts],
            "fatigue_index": round(vbt.fatigue_index, 1),
            "est_set_rpe":   vbt.set_rpe,
        },
        "total_reps": left.counter + right.counter,
    }

    with open("exercise_data.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n[gym_tracker] Saved exercise_data.json")
    print(f"  Total reps   : {output['total_reps']}  (L={left.counter}, R={right.counter})")
    print(f"  Fatigue idx  : {vbt.fatigue_index:.1f}%")
    print(f"  Est. Set RPE : {vbt.set_rpe}")
    print(f"  Dataset rows : {collected_count} appended to {DATASET_FILE}")

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()