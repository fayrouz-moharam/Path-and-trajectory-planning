# APEX ReferenceTrajectory v0.1 (planning -> reference_adapter -> MPC), proposed to Omar 2026-11-08.
# One type for both references: GLOBAL = the whole lap from planning's offline file (T5),
# LOCAL = the stretch ahead of the car from obstacle avoidance (T7).
# Units and signs as planning_reference.csv; centre line = perception's apex_track. No a_ref yet
# (its meaning is still to be agreed with control).

std_msgs/Header header      # stamp = when this reference was made, frame_id = "map"

uint8 SOURCE_GLOBAL = 0     # whole lap, every 5 cm, sent once (latched)
uint8 SOURCE_LOCAL = 1      # 8 m ahead of the car, every 0.25 m, every 0.1 s
uint8 source

string track_id             # e.g. "closed1"
string track_hash           # track it was made for: reject if it differs from the loaded track
string boundary_hash        # walls it was made for
float64 lap_length_m        # lap length [m]
bool valid                  # false = do not drive it
string reason               # why not valid ("" when valid)
uint32 version              # +1 for every new reference

# one entry per point (all arrays the same length)
float64[] s_m               # along the centre line. GLOBAL: 0..lap_length (wraps). LOCAL: continuous s_abs
float64[] x_m               # map position [m]
float64[] y_m
float64[] e_y_ref_m         # where to be: distance from the centre line, + = left [m]
float64[] e_psi_ref_rad     # which way to point, relative to the centre line [rad]
float64[] kappa_ref_1pm     # how much to turn: curvature [1/m], + = left
float64[] v_ref_mps         # how fast [m/s]
float64[] e_y_min_m         # corridor for the car's centre: right edge (walls - margin) [m]
float64[] e_y_max_m         # corridor: left edge [m]
