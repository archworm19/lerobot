# Protocols

How to drive a session on the robot, whoever drives it (the VLM, or Claude Code in its seat): the procedures that
sessions have shown to work. Each says what to do and why, with the session that taught it
([sessions.md](sessions.md)). Rules still being tried are in pen2's "Proposed rules"; they move here once a session
has worked with them.

## 1. Stop at the attack point and check before every grasp

Never plan straight through to contact. Go to the **attack point** (the pose just before the grasp, a few cm back
along the gripper axis), hold there, and before going in:

1. **Look at the grasp:** point the camera at the planned grasp point (`look_at`, head only; the arm keeps holding)
   and capture.
2. **Re-locate the object** in that capture and compare with where the plan had it. Moved more than ~0.5 cm, or not
   found: stop and ask.
3. **Measure the arm:** where the gripper really is (the model at the measured joints) against the plan.
4. **Check the move in:** the straight move along the gripper axis, from where the arm really is, must enter the
   grasp with only the object between the open jaws, and the gripper's body clear of everything else (what the object
   stands on or in, its neighbors).
5. **Re-solve the grasp from this capture**, aimed to cancel the measured error, then go in and close.

If a check cannot be made (the jaws or the object hidden, no depth on it) or fails, do not go in: clear the arm and
look again, or re-plan the attack (higher, another pitch, another side, the other arm).

**Why.** The arm is not accurate: it lands 0.5-1.5 cm from plan (a ~1.2 cm sag at a level reach), the size of a thin
object or a clearance, and the head camera cannot see a grasp well from where the scan looked. Every pen2 failure
was a move into contact that nothing had checked (the pen nudged three times, knocked off a rim, a book pushed by the
gripper on the way in); container1, checked this way, worked.
