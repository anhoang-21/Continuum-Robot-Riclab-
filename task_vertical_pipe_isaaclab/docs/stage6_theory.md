# Stage 6: theory notes (uncertainty, identifiability, pre-tension, safety filter, constrained RL)

Notation: q ∈ R^30 bending joint angles (chain order), u ∈ R^6 motor-nut positions (M1 M2 | M5 M6 | M3 M4),
G ∈ R^{12×30} = ∂l/∂q the cable-length Jacobian of the routing (`cables.CableGeometry`), S ∈ {0, ±1}^{12×6}
the pair map (cable a of motor m pays out +u_m, cable b −u_m), k_c cable stiffness, T0 pre-tension,
k_j backbone stiffness per joint axis, τ_g(q) gravity (and payload) torques.

## 1. Cable drive under uncertainty (`uncertain_cables.py`)

Frictionless law (`cables.CableModel`): T = k_c max(0, Gq − Su + T0/k_c) (+ damping), joint torques −Gᵀ T.
Static balance r(q, u) = −Gᵀ T − k_j q + τ_g(q) = 0.

Capstan friction (the element of `home_hysteresis.py`): the cable is a nut-side spring k_n = k_c/φ, a friction
element at payout p and a robot-side spring k_r = k_c/(1−φ):

    T_n = k_n max(0, p − s + T0/k_n),   T_r = k_r max(0, g − p + T0/k_r),   s = Su, g = Gq.

Stick: p constant while e^{−μθ} ≤ T_n/T_r ≤ e^{μθ}. Slide on the bound λ ∈ {e^{±μθ}}:

    T_n = λ T_r   ⇒   T_r = κ (T0 + k_c (g − s)),   κ = (k_n + k_r) / (k_n + λ k_r),

so a sliding cable obeys the frictionless law scaled by κ (κ = 1 for μθ = 0: the model reduces exactly to
`CableModel`, which `tests/check_robust_env.py` checks bit for bit). After a reversal of the nut the
element sticks until T_n has crossed the band: a dead band of nut travel

    Δs = (λ − 1/λ) T_r / k_n        (checked: 0.640 mm simulated, 0.648 mm predicted at μθ = 0.3, T_r ≈ 3 N).

Nut backlash b: play operator x_n = clip(x_n, x − b/2, x + b/2) between the encoder x and the nut x_n.

## 2. The pre-tension is invisible in the shape (identifiability, ND1)

For an antagonistic pair the routing is mirrored: cable b runs through the holes opposite cable a, so at the
linearisation point G_b = −G_a. With both cables taut

    T_a = k_c (G_a q − u_m) + T0,     T_b = k_c (−G_a q + u_m) + T0,
    torque on the arm  −G_aᵀ T_a − G_bᵀ T_b = −G_aᵀ (T_a − T_b) = −2 k_c G_aᵀ (G_a q − u_m),

independent of T0. **While every cable is taut, the static shape does not depend on the pre-tension at all**:
encoders + shape measurements (mocap, IMUs) cannot identify T0 (or its pair asymmetry); it only shows where a
cable goes slack, and directly in a load cell (T_a + T_b = 2 T0). This is the model-side reason for the load
cells of P0.1, and `sysid.py --study` measures it (T0 error with / without load cells).

## 3. Pre-tension and the tension band (ND3 / ND4)

With both cables of a pair taut T_a + T_b = 2 T0 (from the two laws above). Holding a bend needs a pair
differential D = |T_a − T_b| (the torque balance fixes D, not the sum). Requiring T_min ≤ T ≤ T_max for both:

    D ≤ D_max(T0) = min(2 (T0 − T_min), 2 (T_max − T0)),    maximal at  T0* = (T_min + T_max) / 2.

For the band [0.5, 10] N: T0* = 5.25 N, D_max = 9.5 N. The insertion S-curves need D up to 9.1 N (95th
percentile), so with today's 3 N (D_max = 5 N) the inner cable of a bent section goes slack.
`pretension_study.py` (300 task poses): 41.7 % of the poses keep every cable in the band at 3 N, 96.3 % at
5.25 N (92.6 % for robots of the train uncertainty set), and the curve is symmetric about T0* as the bound
says. With one motor per pair and fixed springs, T0 is a design constant; a motorised tension box (P2) makes
it a control input (raise T0 for strong bends, lower it to reduce friction and wear).

## 4. Tension sensitivity (`safety_filter.py`)

Differentiating the static balance at fixed active set (taut cables, K_a = diag(k_c [taut])) and with the
gravity stiffness K_g taken at the straight hanging pose:

    H dq = Gᵀ K_a S du,   H = H0 + Gᵀ K_a G,   H0 = k_j I − K_g
    G dq/du = G (H0 + Gᵀ K_a G)^{-1} Gᵀ K_a S = P (I + K_a P)^{-1} K_a S,    P = G H0^{-1} Gᵀ  (12 × 12)

(push-through identity: (H0 + Gᵀ K G)^{-1} Gᵀ = H0^{-1} Gᵀ (I + K G H0^{-1} Gᵀ)^{-1}), and for the virtual
tensions T_v = k_c (Gq − Su + T0/k_c) (negative = depth of slack)

    A = dT_v/du = diag(k_c) (G dq/du − S).

Against finite differences of the full static model (pose-dependent gravity): median error 0.5 %, 95th
percentile 4 %, max 12 % on taut cables (`tests/check_safety_filter.py`); without K_g the 95th percentile is 17 %.

## 5. Discrete-time robust CBF on the tension band

Barrier functions per cable: h_lo = T − T_min, h_hi = T_max − T; safe set C = {h_lo ≥ 0, h_hi ≥ 0}.
Discrete-time CBF condition (Agrawal & Sreenath, RSS 2017) with 0 < γ ≤ 1:

    h(x_{k+1}) ≥ (1 − γ) h(x_k).

If it holds at every step and h(x_0) ≥ 0 then h(x_k) ≥ (1 − γ)^k h(x_0) ≥ 0 (forward invariance of C); if
h(x_0) < 0 the violation shrinks geometrically, |h(x_k)| ≤ (1 − γ)^k |h(x_0)|.

Robust version. The filter predicts T_{k+1} = T_k + A Δu; the true change is A Δu + w with
|w_c| ≤ ρ_A Σ_m |A_cm| |Δu_m| (relative sensitivity error ρ_A, e.g. the identified k_c interval) — so with
|Δu_m| ≤ u_rate_m, |w_c| ≤ m_c := ρ_A Σ_m |A_cm| u_rate_m. Imposing on the prediction

    T_lo + A Δu − m − T_min ≥ (1 − γ) (T_lo − T_min)     ⇔   A Δu ≥ −γ (T_lo − T_min) + m
    T_max − T_hi − A Δu − m ≥ (1 − γ) (T_max − T_hi)     ⇔  −A Δu ≥ −γ (T_max − T_hi) + m

guarantees the CBF condition for the true system for every admissible w, provided the estimates bound the
true tension (T_lo ≤ T ≤ T_hi). With load cells on the nuts: T_lo = T_n e^{−μ̄θ} − δ, T_hi = T_n e^{μ̄θ} + δ
(the capstan element keeps the robot side within e^{±μθ} of the nut side; δ: load-cell error). Without load
cells: the model's virtual tension ± (δ + ρ_k |T_v − T0|) (identified-parameter error).

The filter step is the QP

    min_a ½ |a − a_ref|² + ρ/2 |s|²   s.t.  C a + s ≥ d,  s ≥ 0,  lo ≤ a ≤ hi   (a = Δu / u_rate),

solved by dual projected gradient with FISTA momentum: for the separable box the inner minimiser is
a(λ) = clip(a_ref + Cᵀ λ, lo, hi), the dual gradient d − C a(λ) − λ/ρ, the step 1/(λ_max(CᵀC) + 1/ρ).
The slack s makes it always solvable; s > 0 means **no motor step satisfies every barrier** (recursive
feasibility does not hold in general: by section 3, at T0 = 3 N many S-curves cannot be held with every cable
in the band). The filter then returns the least-violating step and reports s, which is itself a safety signal.

## 6. Constrained RL (PPO-Lagrangian, `ppo_lag.py`)

max_π J_r(π) s.t. J_ci(π) ≤ d_i with per-step costs c_slack = Σ_c max(0, T_min − T_r,c)/T_min,
c_over = Σ_c max(0, T_c − T_max)/T_max, c_contact = max(0, F_pipe − F_max)/F_max. Lagrangian
L = J_r − Σ_i λ_i (J_ci − d_i); policy step on the PPO-clip surrogate of (Â_r − Σ λ_i Â_ci)/(1 + Σ λ_i) with
separately normalised GAE advantages and a cost critic per constraint; multiplier step by the PID Lagrangian
(Stooke et al., ICML 2020): λ = max(0, K_p e + K_i Σ e + K_d max(0, ΔJ_c)), e = J_c − d.

## 7. Online adaptation (`rma.py`, `shape_control.AdaptiveController`)

RMA (Kumar et al., RSS 2021), explicit form: a teacher π(o, e) sees the normalised parameters e; an
adaptation module e^ = φ(o_{t−49..t}) is fitted by regression on the states the deployed student visits
(DAgger). Its per-parameter R² is a measure of what the motion reveals about the robot (pre-tension, by
section 2, mostly through slack events or load cells).

Shape control: inverse statics u = F^{-1}(b_ref; θ^) plus PI on the measured bends through the inverse
sensitivity M^{-1} (M = ∂b/∂u), with θ^ = (log k_c, T0, k_j) adapted by a damped Gauss-Newton step on the
noise-normalised residuals of the shape (and of the load cells when present), projected to physical ranges.
For a static map locally linear in θ the update is a normalised gradient estimator (Ioannou & Sun, Robust
Adaptive Control, ch. 4): bounded, and convergent under persistent excitation; again T0 needs the tension
residual.
