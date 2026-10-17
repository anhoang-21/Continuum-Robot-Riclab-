"""
==============================================================================
motors.py - the lead-screw cable motors and the elevator motor of the real robot
==============================================================================
Hardware (graduation thesis, Table 4-1 and ch. 3.5 / 4.4):
  * Ezi-SERVO closed-loop steppers with encoder:
        EzM-42M  holding torque 0.44 N m, rotor inertia  54 g cm^2, 1.2 A/phase
        EzM-56M  holding torque 1.00 N m, rotor inertia 280 g cm^2, 3.0 A/phase
    The Z axis uses the EzM-56M; Motor 3 has the "oversized housing" (-> EzM-56M), the
    other five cable motors are EzM-42M.
  * every cable motor turns a 400 mm stainless lead screw with an 8 mm lead (eq. 3.33:
    "one full 360 deg motor turn moves the lead screw nut by exactly 8 mm"); the nut
    winds one cable of an antagonistic pair in and pays the other out.
  * Z axis: T8 lead screw, 8 mm lead, dual 8 mm guide rods.

Model (per 2 ms physics step, sub-integrated at 0.5 ms), on the nut coordinate x (m):
    m_eff x'' = F_motor + (T_a - T_b)          m_eff = J_total (2 pi / lead)^2
    tau = PID(rotor angle error) clipped to the pull-out torque tau_max(rpm)
Lead-screw efficiency from the thread geometry (lead angle, friction), not measured:
    eta = tan(l) (1 - mu tan(l)) / (tan(l) + mu),  eta_back = (tan(l) - mu) / (tan(l) (1 + mu tan(l)))
    motor torque tau = F lead / (2 pi eta) when the motor drives the load, F lead eta_back / (2 pi)
    when the load drives the screw. Written as a thread friction proportional to the axial load,
        tau / k = c0 F + c1 |F| sign(v),  k = lead / 2 pi,  c0,1 = (1 / eta +- eta_back) / 2,
    integrated as implicit Coulomb friction: at standstill the nut sticks while the torque is inside
    the friction band [F k eta_back, F k / eta] (a screw holds its load with less torque than it needs
    to lift it). Switching the efficiency on the sign of v instead makes the position loop chatter.
The pull-out torque curve is the generic shape of a 24 V hybrid stepper (Ezi-Servo catalogue
curves are of this shape); only the holding torque is from the thesis.
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

G = 9.81


@dataclass(frozen=True)
class MotorSpec:
    name: str
    holding_torque: float          # N m
    rotor_inertia: float           # kg m^2
    rated_current: float           # A / phase
    max_rpm: float = 3000.0
    # pull-out torque / holding torque over speed (rpm), generic 24 V hybrid-stepper shape
    curve_rpm: tuple = (0.0, 300.0, 600.0, 1200.0, 2000.0, 3000.0)
    curve_ratio: tuple = (1.0, 0.95, 0.85, 0.62, 0.38, 0.15)

    def pullout_torque(self, rpm):
        return self.holding_torque * np.interp(np.abs(rpm), self.curve_rpm, self.curve_ratio)


EZM_42M = MotorSpec("EzM-42M", holding_torque=0.44, rotor_inertia=54e-7, rated_current=1.2)
EZM_56M = MotorSpec("EzM-56M", holding_torque=1.00, rotor_inertia=280e-7, rated_current=3.0)


@dataclass(frozen=True)
class LeadScrew:
    lead: float = 0.008            # m per revolution (thesis eq. 3.33)
    diameter: float = 0.008        # m, T8
    length: float = 0.400          # m (six 400 mm screws, ch. 4.4)
    friction: float = 0.20         # steel screw / nut, lubricated
    nut_mass: float = 0.03         # kg, nut + bracket + cable ends

    @property
    def lead_angle(self):
        return math.atan(self.lead / (math.pi * (self.diameter - 0.5 * self.lead / 4)))   # at the pitch diameter

    @property
    def efficiency(self):
        t = math.tan(self.lead_angle)
        return t * (1.0 - self.friction * t) / (t + self.friction)

    @property
    def back_efficiency(self):
        t = math.tan(self.lead_angle)
        return max(0.0, (t - self.friction) / (t * (1.0 + self.friction * t)))

    @property
    def friction_coefficients(self):
        """(c0, c1) of tau / k = c0 F + c1 |F| sign(v): the two efficiencies as a load-proportional friction."""
        return 0.5 * (1.0 / self.efficiency + self.back_efficiency), 0.5 * (1.0 / self.efficiency - self.back_efficiency)

    @property
    def inertia(self):
        m = 7900.0 * math.pi * (0.5 * self.diameter) ** 2 * self.length
        return 0.5 * m * (0.5 * self.diameter) ** 2


CABLE_SCREW = LeadScrew()
ELEVATOR_SCREW = LeadScrew(length=0.400)
# cable motors in the env's order M1 M2 M5 M6 M3 M4 (cables.MOTOR_NAMES)
CABLE_MOTOR_SPECS = (EZM_42M, EZM_42M, EZM_42M, EZM_42M, EZM_56M, EZM_42M)
ELEVATOR_MOTOR = EZM_56M
COUPLING_INERTIA = 0.3e-6          # kg m^2, shaft coupling / timing pulleys
ELEVATOR_MOVING_MASS = 12.0 + 0.6  # kg, Elevator_Link (MJCF) + the continuum arm
ELEVATOR_GUIDE_FRICTION = 5.0      # N, LM8UU / LM10UU linear bearings


def screw_torque(force, velocity, screw: LeadScrew):
    """
    Motor torque (N m) that makes the nut exert `force` (N) while it moves at `velocity` (m/s); at
    standstill the torque to start moving in the direction of the force (driving efficiency).
    """
    k = screw.lead / (2.0 * math.pi)
    c0, c1 = screw.friction_coefficients
    lib = torch if isinstance(force, torch.Tensor) else np
    direction = lib.where(lib.abs(velocity) > 1e-6, lib.sign(velocity), lib.sign(force))
    return k * (c0 * force + c1 * lib.abs(force) * direction)


def breakaway_torque(force, screw: LeadScrew):
    """Torque to start moving the nut against `force` (N): the driving efficiency, worst case."""
    return abs(force) * screw.lead / (2.0 * math.pi) / screw.efficiency


def elevator_torque(drive_force, velocity, screw: LeadScrew = ELEVATOR_SCREW):
    """
    Z-axis motor torque. drive_force: force of the elevator joint drive in the simulation (N, + up),
    which carries inertia and pipe contacts but no weight (the sim has no gravity on the carriage).
    """
    force = drive_force + ELEVATOR_MOVING_MASS * G
    if isinstance(velocity, torch.Tensor):
        force = force + ELEVATOR_GUIDE_FRICTION * torch.sign(velocity)
    else:
        force = force + ELEVATOR_GUIDE_FRICTION * np.sign(velocity)
    return screw_torque(force, velocity, screw)


class CableMotorBank:
    """
    Batched dynamics of the 6 cable motors (closed-loop steppers in position mode driving lead screws).
    State: nut position x (m, = cable-a payout), velocity v, PID integral. `step` advances one
    physics step given the target and the cable tensions on the nut.
    """

    def __init__(self, num_envs, device, specs=CABLE_MOTOR_SPECS, screw: LeadScrew = CABLE_SCREW,
                 bandwidth_hz=60.0, substeps=4, alarm_error_rev=1.0, dtype=torch.float64):
        self.device, self.dtype, self.screw, self.substeps = device, dtype, screw, substeps
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        self.specs = specs
        self.k = screw.lead / (2.0 * math.pi)                                    # m per rad
        self.J = t([s.rotor_inertia + screw.inertia + COUPLING_INERTIA for s in specs])
        self.m_eff = self.J / self.k ** 2 + screw.nut_mass                       # kg, reflected to the nut
        w = 2.0 * math.pi * bandwidth_hz
        self.kp, self.kd, self.ki = self.J * w * w, 2.0 * 0.8 * self.J * w, self.J * w ** 3 / 8.0
        self.hold = t([s.holding_torque for s in specs])
        self.i_max = self.hold / self.ki                                         # anti-windup bound of the integral
        # pull-out torque over speed as a 1-rpm lookup table (one gather instead of a piecewise interpolation)
        rpm = np.arange(int(max(s.max_rpm for s in specs)) + 1, dtype=np.float64)
        self.pullout_table = t(np.stack([s.pullout_torque(rpm) for s in specs], axis=1))   # (rpm, motor)
        self._motor_idx = torch.arange(len(specs), device=device)
        self.alarm_error = alarm_error_rev * 2.0 * math.pi
        self.c0, self.c1 = screw.friction_coefficients
        n = num_envs
        z = lambda: torch.zeros(n, len(specs), dtype=dtype, device=device)  # noqa: E731
        # all state is updated in place (cable_env.py replays the physics step as a CUDA graph)
        self.x, self.v, self.integral = z(), z(), z()
        self.torque, self.load, self.error = z(), z(), z()                        # last step (N m, N, rad)
        self.alarm = torch.zeros(n, len(specs), dtype=torch.bool, device=device)

    def reset(self, env_ids, x):
        self.x[env_ids] = x.to(self.dtype)
        for buf in (self.v, self.integral, self.torque, self.load, self.error):
            buf[env_ids] = 0.0
        self.alarm[env_ids] = False

    def pullout(self, v):
        """Pull-out torque (N m) of every motor at nut speed v (m/s)."""
        rpm = torch.clamp(torch.abs(v) / self.screw.lead * 60.0, max=self.pullout_table.shape[0] - 1)
        return self.pullout_table[rpm.round().long(), self._motor_idx]

    def step(self, dt, x_start, x_end, load_fn):
        """
        Advance by dt while the target moves linearly from x_start to x_end.
        load_fn(x, v) -> force of the cables on the nuts (T_a - T_b) (N, 6).
        self.torque / self.load: mean motor torque / nut load over dt.
        """
        h = dt / self.substeps
        tau_sum = torch.zeros_like(self.x)
        load_sum = torch.zeros_like(self.x)
        for s in range(self.substeps):
            target = x_start + (x_end - x_start) * ((s + 1) / self.substeps)
            err = (target - self.x) / self.k                                       # rotor angle error (rad)
            lim = self.pullout(self.v)
            # anti-windup: integrate only while the output is not saturated, integral term <= holding torque
            tau_pd = self.kp * err - self.kd * self.v / self.k
            integral = self.integral + err * h
            free = torch.abs(tau_pd + self.ki * integral) < lim
            self.integral.copy_(torch.where(free, torch.maximum(torch.minimum(integral, self.i_max), -self.i_max),
                                            self.integral))
            tau = tau_pd + self.ki * self.integral
            tau = torch.maximum(torch.minimum(tau, lim), -lim)
            load = load_fn(self.x, self.v)
            # tau / k = c0 F + c1 |F| sign(v): drive force tau / (k c0) and a thread friction (c1 / c0) |F|,
            # |F| ~ |load|, integrated implicitly (stick when the free motion cannot overcome it)
            friction = self.c1 / self.c0 * torch.abs(load)
            v_free = self.v + h * (tau / (self.k * self.c0) + load) / self.m_eff
            dv = h * friction / self.m_eff
            self.v.copy_(torch.where(torch.abs(v_free) <= dv, torch.zeros_like(v_free), v_free - torch.sign(v_free) * dv))
            self.x.add_(self.v * h)
            tau_sum += tau
            load_sum += load
            self.error.copy_(err)
        self.torque.copy_(tau_sum / self.substeps)
        self.load.copy_(load_sum / self.substeps)
        self.alarm |= torch.abs(self.error) > self.alarm_error
        return self.x
