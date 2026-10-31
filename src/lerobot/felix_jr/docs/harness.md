

Replaces scene_tools.md


Research questions: what is the ideal harness / tool set to enable a vlm to accomplish quasi static pick and place tasks. Implied assumption = contact rich manipulation tasks would require finetuning and are beyond the scope of this project.


Research Background:
Voxposer is the seminal work for this project. The one limitation of voxposer is that it was done before foundational llms had strong language support. We're focused on follow-up work that takes advantage of strong vision fundamentals of chat gpt or claude.


Direct Visual Costmaps & Spatial Prompting
PIVOT: Iterative Visual Prompting for Robotic Manipulation (Nasiriany et al., 2024)

How it builds on VoxPoser: Instead of prompting an LLM to write code for a 3D voxel grid, PIVOT feeds RGB images directly into a native VLM (GPT-4V). It visualizes proposed trajectories or target candidate coordinates directly as visual overlays (prompt annotations) on the image. The VLM iteratively evaluates these visual points to select the optimal grasp, trajectory, or placement goal, effectively computing a visual value map natively inside the VLM.

Key advantage over VoxPoser: Bypasses complex 3D voxel code generation by letting the VLM "see" and evaluate motion trajectories directly in image space.

TODO: yes, this seems like a good place to build from. However, we need more information. How strict was the harness / loop? Did they allow the VLM to control how the image was re-rendered? Did they hook the VLM up to a motion planner?

Code-as-Costmaps: Zero-Shot Visuomotor Planning with Vision-Language Models (2024)

How it builds on VoxPoser: Replaces VoxPoser's multi-step visual API pipeline with direct multimodal prompting. Modern VLMs accept the raw workspace RGB image along with high-level language, enabling them to directly output 2D spatial costmap functions or dense coordinate arrays tailored to the visual context. These 2D/2.5D costmaps are then unprojected via depth maps into 3D cost fields for trajectory optimizers (like MPPI or TrajOpt).

Key advantage over VoxPoser: Eliminates errors caused by misaligned bounding boxes or separate detection tools, as the VLM reasons about object geometry and obstacles simultaneously.

TODO: connection to a trajectory optimizer seems like the correct move. Do they make depth information available to the VLM? Do they render proposed trajectories from the optimizer for the VLM?

2. VLM-Guided Spatial Affordances & Motion Primitives
ManipVQA: Injecting Visual Spatial Reasoning into VLMs for Robotic Manipulation (Liu et al., 2024)

How it builds on VoxPoser: VoxPoser treated the VLM as a code synthesizer and relied on heuristics to construct 3D affordance boundaries. ManipVQA fine-tunes native VLMs to output explicit spatial keypoints (e.g., approach vector, contact point, post-grasp trajectory) directly onto the image space, which are projected into metric 3D point clouds for downstream IK and motion planning.   
Medium

Key advantage over VoxPoser: Yields higher precision on complex, asymmetric objects where generic voxel bounding grids fail to capture accurate contact orientations.

TODO: we're not finetuning

RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics (Yuan et al., 2024)

How it builds on VoxPoser: Simplifies the continuous 3D value map framing of VoxPoser into direct, dense visual point predictions. RoboPoint queries a multimodal VLM (Vicuna + vision backbones) directly with spatial language (e.g., "Where should I place the mug to avoid the spill?"). The model outputs exact pixel point affordances 
(u,v) which, when combined with depth images, define exact metric target coordinates 
(X,Y,Z) for traditional path planners (e.g., MoveIt).

Key advantage over VoxPoser: Significantly faster execution time with reduced compute overhead compared to evaluating continuous 3D voxel fields.

TODO: did they finetune for pointing? Seems like that's asking a lot of precision from the VLM? I'm a bit more convinced by the approaches that project a world coordinate grid into camera frame --> ask the VLM to select grid positions like is shown in the recent gemini argon demonstration. Are there any comparisons of these methods?

3. Closed-Loop VLM Trajectory Optimization & Replanning
Sketch-MoMa: Teleoperation and Autonomous Planning for Mobile Manipulators via VLM Sketching (2025)

How it builds on VoxPoser: Uses VoxPoser's framing of framing manipulation as trajectory optimization guided by cost fields. It employs native multimodal VLMs to draw/annotate keyframe trajectories directly onto visual camera frames, which are then passed to continuous velocity or MPC planners.

Key advantage over VoxPoser: Handles whole-body movement and base placement for mobile manipulators far better than localized voxel grids.   
arXiv

TODO: what is meant by key frame trajectories 


This doc does 3 things:
0. summarizes relevant research
1. discusses design principles for the harness
2. records results of experiments; TODO: maybe this should be in a separate file?