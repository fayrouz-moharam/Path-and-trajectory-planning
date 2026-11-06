"""T7 Half B: the way past the cones (Fayrouz). Half A (Mohamed, t7_avoidance/) hands over KeepOuts every cycle.

    zones -> grid.py (B2: corridor grid, disparity-style widening) -> dp.py (B3: Liniger's cheapest path)
          -> detour.py (B4: bent racing line by graph / sampling) -> plan.py (B4 + B5: CyclePlan, checks)

plan.plan_cycle() is the one call per MPC cycle. numpy + scipy only, no ROS.
"""
