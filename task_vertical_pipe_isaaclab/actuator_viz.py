"""
==============================================================================
actuator_viz.py - animated drawing of the cable drive for the videos
==============================================================================
One module per lead-screw motor, as on the real robot (cable_env.py / motors.py):

    motor (dial turns with the rotor angle)  ->  lead screw (thread scrolls as it turns)
    ->  nut (moves 8 mm per turn)  ->  cable a wound in / cable b paid out  ->  to the robot

Drawn from the simulated state of one env: nut position and speed (motor rpm, rotor angle =
nut position / lead), motor torque, cable tensions (line width; grey = slack). Colours per
section = the colours of the cables drawn on the robot in Isaac Sim.
==============================================================================
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import cables as CB
import motors as MT

# BGR, same hues as cable_env.CABLE_COLORS (s1 orange, s2 green, s3 blue, slack grey)
SECTION_BGR = ((38, 102, 242), (89, 204, 51), (255, 140, 64))
SLACK_BGR = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


def _text(img, t, x, y, s=0.45, c=(235, 238, 242), th=1):
    cv2.putText(img, t, (int(x), int(y)), FONT, s, c, th, AA)


def draw_actuators(env, i, width=640, height=720, dark=True, title=True):
    """BGR image (height, width, 3) of the 6 lead-screw modules of env i."""
    bg, ink, sub = ((26, 24, 20), (235, 238, 242), (160, 172, 190)) if dark else \
        ((250, 250, 250), (40, 40, 40), (110, 110, 110))
    img = np.full((height, width, 3), bg, np.uint8)
    x_nut = env.motor[i].detach().cpu().numpy()
    v_nut = env.motors.v[i].detach().cpu().numpy()
    tau = env.motors.torque[i].detach().cpu().numpy() * 1000.0
    T = env.tension[i].detach().cpu().numpy()
    lead = MT.CABLE_SCREW.lead
    rpm = v_nut / lead * 60.0
    angle = x_nut / lead * 2.0 * math.pi                     # rotor angle (rad) = nut travel / lead

    y0 = 28                                                    # room for a caption drawn over the top edge
    if title:
        _text(img, "CABLE DRIVE: motor turns the lead screw, the nut pulls the cable", 14, 26, 0.52, (0, 230, 255))
        _text(img, "Ezi-Servo EzM-42M / 56M  |  8 mm lead screw  |  1 nut = 1 cable pair (a in, b out)",
              14, 48, 0.4, sub)
        y0 = 58
    compact = height < 500                                     # small inset: numbers kept to rpm + torque
    col_w = width / 6.0
    motor_top = y0 + (20 if compact else 26)
    motor_h = int(0.075 * height) if not compact else 30
    screw_top = motor_top + motor_h + 4
    screw_bot = int(height * 0.66) if not compact else height - 84
    sheath_y = screw_bot + 18                                  # the cables enter their Bowden sheaths here
    screw_len = screw_bot - screw_top
    pitch_px = 14.0
    for m in range(6):
        s = CB.MOTOR_SECTION[m]
        col = SECTION_BGR[s]
        cx = int(col_w * (m + 0.5)) - 8
        # motor block + rotating dial
        mw = int(min(col_w * 0.62, 70))
        cv2.rectangle(img, (cx - mw // 2, motor_top), (cx + mw // 2, motor_top + motor_h), (70, 72, 78), -1, AA)
        cv2.rectangle(img, (cx - mw // 2, motor_top), (cx + mw // 2, motor_top + motor_h), col, 2, AA)
        r = int(motor_h * 0.36)
        dc = (cx, motor_top + motor_h // 2)
        cv2.circle(img, dc, r, (200, 204, 210), 1, AA)
        a = angle[m]
        tip = (int(dc[0] + r * math.sin(a)), int(dc[1] - r * math.cos(a)))
        cv2.line(img, dc, tip, (255, 255, 255), 2, AA)
        if abs(rpm[m]) > 3.0:                                  # rotation arrow
            start = -90 + math.degrees(a)
            sweep = 70 if rpm[m] > 0 else -70
            cv2.ellipse(img, dc, (r + 5, r + 5), 0, start, start + sweep, (0, 230, 255), 2, AA)
        _text(img, CB.MOTOR_NAMES[m], cx - mw // 2, motor_top - 6, 0.42 if compact else 0.5, col, 2)
        if not compact:
            _text(img, MT.CABLE_MOTOR_SPECS[m].name[3:], cx + 2, motor_top - 8, 0.33, sub)
        # lead screw with a thread that scrolls with the rotor angle
        sw = 9
        cv2.rectangle(img, (cx - sw // 2, screw_top), (cx + sw // 2, screw_bot), (120, 124, 132), -1)
        phase = (a / (2.0 * math.pi)) * pitch_px
        k0 = int(-phase // pitch_px) - 1
        for k in range(k0, k0 + int(screw_len / pitch_px) + 3):
            y = screw_top + phase % pitch_px + (k - k0) * pitch_px - pitch_px
            if screw_top <= y <= screw_bot - 5:
                cv2.line(img, (cx - sw // 2, int(y)), (cx + sw // 2, int(y + 5)), (200, 204, 212), 1, AA)
        # nut: centre of the screw = 0 mm, up = cable a wound in (negative displacement)
        rng = CB.MOTOR_RANGE[m]
        ny = int((screw_top + screw_bot) / 2 + (x_nut[m] / rng) * 0.5 * (screw_len - 30))
        nw, nh = (24, 14) if compact else (30, 16)
        cv2.rectangle(img, (cx - nw // 2, ny - nh // 2), (cx + nw // 2, ny + nh // 2), col, -1, AA)
        cv2.rectangle(img, (cx - nw // 2, ny - nh // 2), (cx + nw // 2, ny + nh // 2), (20, 20, 20), 1, AA)
        # the two cables of the pair leave the nut down into their sheaths (to the robot): nut up = cable a in
        for c_idx, dx in ((2 * m, 1), (2 * m + 1, -1)):
            t = T[c_idx]
            slack = t < 0.05
            thick = 1 if slack else int(np.clip(1 + t / 3.0, 1, 5))
            ccol = SLACK_BGR if slack else col
            px = cx + dx * (nw // 2)
            pts = np.array([[px, ny], [px + dx * 10, ny], [px + dx * 10, sheath_y]], np.int32)
            cv2.polylines(img, [pts], False, ccol, thick, AA)
            _text(img, "a" if dx > 0 else "b", px + dx * 10 - 4, sheath_y + 13, 0.35, ccol)
        _text(img, f"{x_nut[m] * 1000:+5.1f}", cx - (16 if compact else 20), ny + 4, 0.3 if compact else 0.36,
              (255, 255, 255))
        if compact:                                            # rpm + torque only
            _text(img, f"{rpm[m]:+4.0f}rpm", cx - 26, sheath_y + 32, 0.33, ink)
            _text(img, f"{tau[m]:+4.1f}mNm", cx - 26, sheath_y + 50, 0.33, ink)
            continue
        # numbers under the sheaths
        yb = sheath_y + 36
        _text(img, f"{rpm[m]:+5.0f} rpm", cx - 34, yb, 0.4, ink)
        _text(img, f"{tau[m]:+5.1f} mNm", cx - 34, yb + 20, 0.4, ink)
        ta, tb = T[2 * m], T[2 * m + 1]
        _text(img, f"a {ta:4.1f} N", cx - 34, yb + 40, 0.38, SLACK_BGR if ta < 0.05 else col)
        _text(img, f"b {tb:4.1f} N", cx - 34, yb + 58, 0.38, SLACK_BGR if tb < 0.05 else col)
    cv2.line(img, (8, sheath_y), (width - 8, sheath_y), (90, 94, 102), 1, AA)
    if compact:
        return img
    _text(img, "Bowden sheaths -> 12 cables through the discs of the arm", 14, height - 42, 0.4, sub)
    # legend
    yl = height - 16
    x = 14
    for s, name in enumerate(("section 1 (d 40 mm)", "section 2 (30 mm)", "section 3 (20 mm)")):
        cv2.rectangle(img, (x, yl - 10), (x + 16, yl), SECTION_BGR[s], -1)
        _text(img, name, x + 22, yl, 0.38, sub)
        x += 22 + 8 * len(name) + 18
    cv2.line(img, (x, yl - 5), (x + 16, yl - 5), SLACK_BGR, 1, AA)
    _text(img, "slack", x + 22, yl, 0.38, sub)
    return img
