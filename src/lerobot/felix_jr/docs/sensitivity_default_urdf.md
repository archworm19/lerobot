# URDF sensitivity report

URDF: `src/lerobot/felix_jr/xlerobot.urdf` (default, no RealSense mount yet), left arm, camera link `head_camera_link` (fx 611.0 px, 640x480). 54 configurations: head tilt 40-69 deg down, pan -14..+14 deg, random arm poses with the jaw pivot in view.

Jaw pivot distance from the camera: 0.16 to 0.51 m (median 0.35). Visible fraction per target: elbow 15%, wrist 61%, jaw pivot 100%

## Assumed tolerances (1 sigma, guesses)

| Kind | Sigma |
|---|---|
| joint zero | 2.0 deg |
| joint origin position | 2 mm |
| head tower position | 3 mm |
| head tower rotation | 1.0 deg |
| arm mount position | 3 mm |
| arm mount rotation | 1.0 deg |
| camera mount position | 5 mm |
| camera mount rotation | 2.0 deg |

## Monte Carlo: total pixel error with no calibration

| Parameters perturbed | Target | Mean px | 95th pct px | Max px |
|---|---|---|---|---|
| head only | elbow | 47.4 | 97.3 | 154.6 |
| head only | wrist | 45.6 | 90.2 | 172.1 |
| head only | jaw pivot | 45.4 | 88.6 | 181.8 |
| arm only | elbow | 19.8 | 41.4 | 83.4 |
| arm only | wrist | 23.2 | 50.3 | 191.8 |
| arm only | jaw pivot | 25.8 | 56.6 | 302.1 |
| all | elbow | 53.8 | 108.2 | 152.9 |
| all | wrist | 53.6 | 109.7 | 380.3 |
| all | jaw pivot | 54.9 | 113.3 | 313.4 |

## Parameters ranked by pixel shift of the jaw pivot at their assumed tolerance

| Parameter | Group | px per unit (mean / worst) | px at 1 sigma |
|---|---|---|---|
| `zero:head_tilt_joint` | head | 12.0 / 13.5 px per deg | 24.0 |
| `rot:fixed_head_camera_link:yaw` | head | 11.8 / 13.7 px per deg | 23.6 |
| `rot:fixed_head_camera_link:pitch` | head | 11.2 / 12.3 px per deg | 22.5 |
| `zero:head_pan_joint` | head | 8.2 / 14.2 px per deg | 16.3 |
| `zero:Rotation_L` | arm | 8.1 / 16.2 px per deg | 16.3 |
| `zero:Pitch_L` | arm | 6.2 / 16.9 px per deg | 12.3 |
| `rot:fixed_top_base_link:yaw` | head | 11.4 / 21.4 px per deg | 11.4 |
| `zero:Elbow_L` | arm | 5.3 / 10.9 px per deg | 10.6 |
| `pos:fixed_head_camera_link:y` | head | 1.9 / 3.8 px per mm | 9.5 |
| `pos:fixed_head_camera_link:z` | head | 1.9 / 3.8 px per mm | 9.5 |
| `rot:fixed_Base:yaw` | arm | 8.0 / 14.4 px per deg | 8.0 |
| `rot:fixed_head_camera_link:roll` | head | 3.9 / 6.7 px per deg | 7.8 |
| `pos:fixed_Base:y` | arm | 1.9 / 4.0 px per mm | 5.7 |
| `pos:fixed_top_base_link:y` | head | 1.9 / 4.0 px per mm | 5.7 |
| `rot:fixed_top_base_link:roll` | head | 5.2 / 24.5 px per deg | 5.2 |
| `rot:fixed_top_base_link:pitch` | head | 4.8 / 13.8 px per deg | 4.8 |
| `pos:fixed_Base:x` | arm | 1.6 / 3.0 px per mm | 4.8 |
| `pos:fixed_top_base_link:x` | head | 1.6 / 3.0 px per mm | 4.8 |
| `zero:Wrist_Pitch_L` | arm | 2.3 / 4.9 px per deg | 4.6 |
| `rot:fixed_Base:roll` | arm | 4.3 / 23.4 px per deg | 4.3 |
| `rot:fixed_Base:pitch` | arm | 3.8 / 10.5 px per deg | 3.8 |
| `pos:Rotation_L:y` | arm | 1.9 / 4.0 px per mm | 3.8 |
| `pos:head_pan_joint:y` | head | 1.9 / 4.0 px per mm | 3.8 |
| `pos:head_tilt_joint:y` | head | 1.9 / 3.8 px per mm | 3.8 |
| `pos:Wrist_Roll_L:x` | arm | 1.9 / 3.5 px per mm | 3.7 |
| `pos:Wrist_Pitch_L:x` | arm | 1.9 / 3.5 px per mm | 3.7 |
| `pos:Elbow_L:x` | arm | 1.9 / 3.5 px per mm | 3.7 |
| `pos:Pitch_L:x` | arm | 1.9 / 3.5 px per mm | 3.7 |
| `pos:fixed_top_base_link:z` | head | 1.2 / 3.6 px per mm | 3.6 |
| `pos:fixed_Base:z` | arm | 1.2 / 3.6 px per mm | 3.6 |
| `pos:fixed_head_camera_link:x` | head | 0.7 / 2.0 px per mm | 3.5 |
| `pos:Jaw_L:z` | arm | 1.7 / 3.9 px per mm | 3.4 |
| `pos:Jaw_L:x` | arm | 1.7 / 3.5 px per mm | 3.3 |
| `pos:Pitch_L:z` | arm | 1.7 / 3.0 px per mm | 3.3 |
| `pos:Elbow_L:y` | arm | 1.6 / 3.1 px per mm | 3.3 |
| `pos:head_tilt_joint:x` | head | 1.6 / 3.2 px per mm | 3.2 |
| `pos:Rotation_L:x` | arm | 1.6 / 3.0 px per mm | 3.2 |
| `pos:head_pan_joint:x` | head | 1.6 / 3.0 px per mm | 3.2 |
| `pos:Wrist_Roll_L:z` | arm | 1.5 / 3.1 px per mm | 3.1 |
| `pos:Wrist_Pitch_L:y` | arm | 1.4 / 2.8 px per mm | 2.9 |
| `pos:Wrist_Pitch_L:z` | arm | 1.3 / 3.6 px per mm | 2.7 |
| `pos:Wrist_Roll_L:y` | arm | 1.3 / 3.5 px per mm | 2.6 |
| `pos:Jaw_L:y` | arm | 1.3 / 3.5 px per mm | 2.6 |
| `pos:Pitch_L:y` | arm | 1.2 / 3.6 px per mm | 2.4 |
| `pos:head_pan_joint:z` | head | 1.2 / 3.6 px per mm | 2.4 |
| `pos:head_tilt_joint:z` | head | 1.2 / 3.6 px per mm | 2.4 |
| `pos:Rotation_L:z` | arm | 1.2 / 3.6 px per mm | 2.4 |
| `pos:Elbow_L:z` | arm | 1.2 / 3.3 px per mm | 2.4 |
| `zero:Wrist_Roll_L` | arm | 0.6 / 1.4 px per deg | 1.2 |

## By kind (jaw pivot, mean over parameters of that kind)

| Kind | Group | px per unit (mean) | px at 1 sigma (mean) |
|---|---|---|---|
| joint zero | head | 10.1 px per deg | 20.2 |
| camera mount rotation | head | 9.0 px per deg | 18.0 |
| joint zero | arm | 4.5 px per deg | 9.0 |
| camera mount position | head | 1.5 px per mm | 7.5 |
| head tower rotation | head | 7.1 px per deg | 7.1 |
| arm mount rotation | arm | 5.4 px per deg | 5.4 |
| head tower position | head | 1.6 px per mm | 4.7 |
| arm mount position | arm | 1.6 px per mm | 4.7 |
| joint origin position | head | 1.6 px per mm | 3.1 |
| joint origin position | arm | 1.6 px per mm | 3.1 |
