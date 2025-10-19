#!/usr/bin/env python3
"""
LLM-Guided Robot Arm Pick and Place with MuJoCo and Gemini Vision
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
import google.generativeai as genai
import io
import json

class TrajectoryLogger:
    """Logs trajectory execution for analysis"""
    def __init__(self):
        self.waypoints = []
        self.object_positions_initial = {}
        self.object_positions_final = {}

    def reset(self):
        self.waypoints = []
        self.object_positions_initial = {}
        self.object_positions_final = {}

    def log_waypoint(self, waypoint_idx, planned_ee_pos, actual_ee_pos, gripper_state):
        """Record a waypoint execution"""
        deviation = np.linalg.norm(np.array(planned_ee_pos) - np.array(actual_ee_pos))
        self.waypoints.append({
            "waypoint": waypoint_idx,
            "planned": list(planned_ee_pos),
            "actual": list(actual_ee_pos),
            "gripper": float(gripper_state),
            "deviation": float(deviation)
        })

    def log_object_position(self, object_name, position, is_final=False):
        """Record object position"""
        if is_final:
            self.object_positions_final[object_name] = list(position)
        else:
            self.object_positions_initial[object_name] = list(position)

    def get_structured_state(self, failure_analysis):
        """Get complete structured state for Gemini"""
        return {
            "objects": {
                name: {
                    "initial": self.object_positions_initial.get(name, [0, 0, 0]),
                    "final": self.object_positions_final.get(name, [0, 0, 0])
                }
                for name in set(list(self.object_positions_initial.keys()) + list(self.object_positions_final.keys()))
            },
            "trajectory": self.waypoints,
            "failure_analysis": failure_analysis
        }


class PIDController:
    """Simple PID controller for joint trajectory following"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class GeminiVisionController:
    """Uses Gemini to analyze scene and generate robot trajectories"""
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def analyze_scene(self, image_array, execution_frames=None, structured_state=None):
        """Send image and structured data to Gemini for trajectory planning

        Args:
            image_array: Initial overhead view
            execution_frames: List of 2-3 key images (initial, grasp, final)
            structured_state: Dict with objects, trajectory, failure_analysis
        """
        # Convert numpy array to PIL Image
        images = [Image.fromarray(image_array)]

        # Add execution frames if provided (now just 3 key frames)
        if execution_frames:
            images.extend([Image.fromarray(frame) for frame in execution_frames])

        prompt = """You are controlling a Franka Panda robot arm to pick and place objects.

"""

        # Add structured state data if available (from previous attempt)
        if structured_state:
            prompt += f"""## PREVIOUS ATTEMPT DATA

Objects (before -> after):
{json.dumps(structured_state['objects'], indent=2)}

Trajectory Execution (planned vs actual end-effector positions):
{json.dumps(structured_state['trajectory'], indent=2)}

Failure Analysis:
{json.dumps(structured_state['failure_analysis'], indent=2)}

Images show:
1. Initial workspace view
2. Grasp attempt moment
3. Final state

ANALYZE: What went wrong? Use the numerical data to identify:
- Was the grasp position accurate? (check deviation at waypoint 1)
- Did the robot reach the planned positions? (check all deviations)
- Where did the object end up vs where it should be?

Generate an IMPROVED plan based on this analysis.
"""
        else:
            prompt += """Analyze this overhead camera view of the workspace.
"""

        prompt += """
## Workspace Layout:
- Red and blue cubes (need to be picked up)
- Green circular target zone (where objects should be placed)
- Robot arm base at position (0, 0)
- Camera shows approximately:
  - X axis: left (-0.2) to right (+0.6)
  - Y axis: back (-0.4) to front (+0.4)
  - Z is height above table (0.05 = on table surface)

## Your Task:
Return a JSON object with a pick-and-place plan:
```json
{
  "analysis": "What you observe and what went wrong (if previous attempt data provided)",
  "target_object": "red or blue",
  "object_position": [x, y, z],
  "target_position": [x, y, z],
  "grasp_approach": [x, y, z],
  "place_approach": [x, y, z],
  "confidence": 0.0 to 1.0
}
```

**Important:**
- Pick ONE object at a time
- Approach from above (higher Z), then descend
- Use precise coordinates from structured data if available
- If previous attempt failed due to grasp offset, adjust grasp_approach position accordingly"""

        try:
            # Send prompt and all images to Gemini
            content = [prompt] + images
            response = self.model.generate_content(content)
            text = response.text.strip()

            # Extract JSON from response (it might have markdown formatting)
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text

            plan = json.loads(json_str)
            return plan
        except Exception as e:
            print(f"Error analyzing scene: {e}")
            # Return a default plan if Gemini fails
            return {
                "analysis": "Failed to analyze, using default",
                "target_object": "red",
                "object_position": [0.5, 0.2, 0.025],
                "target_position": [0.3, -0.3, 0.05],
                "grasp_approach": [0.5, 0.2, 0.15],
                "place_approach": [0.3, -0.3, 0.15],
                "confidence": 0.3
            }

    def ask_next_action(self, current_image, trajectory_history, object_states, target_object, target_zone_pos):
        """Ask Gemini for the next single action/waypoint to execute

        Args:
            current_image: Current overhead camera view
            trajectory_history: List of previously executed waypoints with results
            object_states: Current object positions and states
            target_object: Which object we're trying to grasp
            target_zone_pos: Where we want to place the object

        Returns:
            Dict with: action (approach/grasp/lift/move/place/release/retreat/done),
                      target_pos [x,y,z], gripper (0.0-0.04), reasoning
        """
        img = Image.fromarray(current_image)

        # Build trajectory history string
        history_str = ""
        if trajectory_history:
            history_str = "## Previous Actions:\n"
            for i, step in enumerate(trajectory_history):
                history_str += f"{i}. {step['action']}: moved to {step['target_pos']}, gripper={step['gripper']}, result={step.get('result', 'unknown')}\n"
        else:
            history_str = "## This is the first action of the trajectory.\n"

        prompt = f"""You are controlling a Franka Panda robot arm ONE STEP AT A TIME.

{history_str}

## Current State:
Object states: {json.dumps(object_states, indent=2)}
Target object: {target_object}
Target zone position: {target_zone_pos}

## Your Task:
Decide the NEXT SINGLE ACTION. Return JSON:
```json
{{
  "action": "approach|grasp|lift|move|place|release|retreat|done|retry_previous",
  "target_pos": [x, y, z],
  "gripper": 0.04 for open or 0.0 for closed,
  "reasoning": "why this action"
}}
```

## Action Meanings:
- approach: Move above object
- grasp: Descend and close gripper
- lift: Lift object up
- move: Move toward target zone
- place: Lower to place position
- release: Open gripper
- retreat: Move up and away
- done: Task complete
- retry_previous: Go back and retry last action with adjustment

## Important:
- Gripper fingers must point DOWN (vertical)
- Heights: table=0.05, safe approach=0.15-0.20
- Only return ONE action
- Be conservative with positions"""

        try:
            response = self.model.generate_content([prompt, img])
            text = response.text.strip()

            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text

            action = json.loads(json_str)
            return action
        except Exception as e:
            print(f"Error getting next action: {e}")
            return {
                "action": "done",
                "target_pos": [0, 0, 0.2],
                "gripper": 0.04,
                "reasoning": "Error occurred, aborting"
            }

    def check_waypoint(self, waypoint_idx, waypoint_data, current_image, object_states):
        """Ask Gemini if trajectory should continue or abort after a waypoint

        Args:
            waypoint_idx: Index of waypoint just completed
            waypoint_data: Dict with planned, actual, deviation, gripper
            current_image: Camera frame after waypoint
            object_states: Dict with object positions and movement status

        Returns:
            Dict with decision: "continue" or "abort", reason, and optional adjustments
        """
        img = Image.fromarray(current_image)

        # Determine what this waypoint was supposed to do
        waypoint_purposes = {
            0: "approach object from above",
            1: "descend to grasp position",
            2: "close gripper to grasp object",
            3: "lift object",
            4: "move object toward target",
            5: "position above target",
            6: "lower to place object",
            7: "open gripper to release object"
        }

        purpose = waypoint_purposes.get(waypoint_idx, "execute motion")

        prompt = f"""You are monitoring a robot arm executing a pick-and-place task in real-time.

## Waypoint Just Completed: #{waypoint_idx} - {purpose}

**Execution Data:**
```json
{json.dumps(waypoint_data, indent=2)}
```

**Object States:**
```json
{json.dumps(object_states, indent=2)}
```

**Critical Questions:**
1. Did this waypoint achieve its intended purpose?
   - Waypoint 2 (close gripper): Did the object get grasped? Check if object moved upward.
   - Waypoint 3 (lift): Is the object still in gripper? Check object height.
   - Waypoint 7 (release): Did object release onto target?

2. Should we continue to the next waypoint or abort and replan?

**Decision Rules:**
- If grasp failed (waypoint 2) → ABORT immediately, don't waste time on remaining waypoints
- If object dropped during lift (waypoint 3) → ABORT
- If significant deviation (>5cm) from planned position → Consider ABORT
- Otherwise → CONTINUE

Return JSON:
```json
{{
  "decision": "continue" or "abort",
  "reason": "Brief explanation of why",
  "observed_issue": "What went wrong (if aborting)",
  "suggested_correction": "How to fix it in next attempt (if aborting)"
}}
```

Be decisive. It's better to abort early and replan than continue with a failed grasp."""

        try:
            response = self.model.generate_content([prompt, img])
            text = response.text.strip()

            # Extract JSON
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text

            decision = json.loads(json_str)
            return decision
        except Exception as e:
            print(f"Error checking waypoint: {e}")
            # Default to continue on error
            return {
                "decision": "continue",
                "reason": "Error in waypoint check, continuing by default",
                "observed_issue": "",
                "suggested_correction": ""
            }


class RobotController:
    """Main robot control system with MuJoCo and Gemini integration"""
    def __init__(self, model_path="franka_panda/pick_place_scene.xml"):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # PID controllers for each joint
        self.n_joints = 7  # Franka has 7 arm joints
        self.pids = [PIDController(kp=10.0, kd=2.0) for _ in range(self.n_joints)]
        self.gripper_pid = PIDController(kp=5.0, kd=1.0)

        # Camera rendering
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")

        # Gemini controller
        self.gemini = GeminiVisionController()

        # Task state
        self.task_phase = "identify_target"  # identify_target, execute_action, check_success
        self.attempt = 0
        self.max_attempts = 5
        self.trajectory_history = []  # List of all executed actions with results
        self.target_object = None  # Which object we're trying to grasp
        self.target_zone_pos = None  # Where to place the object

        # Get end-effector body ID for tracking
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Get target zone position (fixed)
        target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
        target_body_id = self.model.geom_bodyid[target_geom_id]
        self.target_zone_pos = self.data.xpos[target_body_id].copy()

    def reset_simulation(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        # Set joint7 to vertical position (0.0 radians for fingers pointing down)
        self.data.qpos[6] = 0.0
        for pid in self.pids:
            pid.reset()
        self.gripper_pid.reset()
        # Clear trajectory history
        self.trajectory_history = []
        self.target_object = None

    def get_camera_image(self):
        """Capture image from overhead camera"""
        self.renderer.update_scene(self.data, camera=self.camera_id)
        pixels = self.renderer.render()
        return pixels

    def get_object_states(self):
        """Get current state of all objects for waypoint checking"""
        states = {}
        for cube_name in ["red_cube", "blue_cube"]:
            try:
                cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
                cube_pos = self.data.xpos[cube_body_id]
                cube_vel = self.data.cvel[cube_body_id][:3]  # Linear velocity

                # Check if object moved from initial position
                initial_pos = np.array([0.5, 0.2, 0.025]) if cube_name == "red_cube" else np.array([0.4, -0.15, 0.025])
                moved = np.linalg.norm(cube_pos[:2] - initial_pos[:2]) > 0.02

                states[cube_name] = {
                    "position": list(cube_pos),
                    "height": float(cube_pos[2]),
                    "velocity": list(cube_vel),
                    "moved_from_initial": bool(moved),
                    "is_moving": bool(np.linalg.norm(cube_vel) > 0.01)
                }
            except:
                pass
        return states

    def inverse_kinematics_simple(self, target_pos):
        """
        Simple IK using MuJoCo's built-in IK solver.
        Returns joint positions to reach target_pos with end effector.
        Constrains orientation to keep gripper pointing down.
        """
        # Get end effector body ID
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Save current state and lock joint7
        qpos_init = self.data.qpos.copy()

        # Always keep joint7 at 0.0 (vertical, fingers pointing down)
        self.data.qpos[6] = 0.0

        # Target orientation: Z-axis pointing down (for gripper)
        # When gripper points down, the hand's Z-axis should point downward
        target_z_axis = np.array([0, 0, -1])  # Pointing down

        # Iterative IK
        alpha = 0.3  # Step size (reduced for stability)
        tol_pos = 0.01   # Position tolerance (1cm)
        tol_ori = 0.05   # Orientation tolerance (much stricter)
        max_iter = 200   # More iterations for convergence
        ori_weight = 2.0  # Give orientation more importance

        print(f"[IK START] Target pos: {target_pos}, Target Z-axis: {target_z_axis}")

        for i in range(max_iter):
            mujoco.mj_forward(self.model, self.data)

            # Current end effector position
            ee_pos = self.data.xpos[ee_id].copy()

            # Current end effector orientation (rotation matrix)
            ee_mat = self.data.xmat[ee_id].reshape(3, 3)
            current_z_axis = ee_mat[:, 2]  # Z-axis of the hand

            # Position error
            pos_error = target_pos - ee_pos

            # Orientation error: align Z-axis to point down
            # Cross product gives rotation axis scaled by sin(angle)
            ori_error_raw = np.cross(current_z_axis, target_z_axis)
            ori_error = ori_error_raw * ori_weight

            # Combined error (position + weighted orientation)
            error = np.concatenate([pos_error, ori_error])

            if np.linalg.norm(pos_error) < tol_pos and np.linalg.norm(ori_error_raw) < tol_ori:
                print(f"[IK] Converged in {i} iterations - pos_err: {np.linalg.norm(pos_error):.4f}, ori_err: {np.linalg.norm(ori_error_raw):.4f}, Z-axis: {current_z_axis}")
                break

            # Compute Jacobian (position and rotation)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, ee_id)

            # Combine position and rotation Jacobians for first 6 DOFs
            J = np.vstack([jacp[:, :6], jacr[:, :6]])  # 6x6 matrix

            # Damped least squares
            damping = 0.01
            JJT = J @ J.T + damping * np.eye(6)
            delta_q = J.T @ np.linalg.solve(JJT, alpha * error)

            # Update joint positions (only first 6 joints)
            self.data.qpos[:6] += delta_q

            # Keep joint7 locked at 0.0 (vertical)
            self.data.qpos[6] = 0.0

        # Check if IK failed to converge
        if i == max_iter - 1:
            print(f"[IK WARNING] Failed to converge! pos_err: {np.linalg.norm(pos_error):.4f}, ori_err: {np.linalg.norm(ori_error_raw):.4f}")
            print(f"[IK WARNING] Final Z-axis: {current_z_axis}, target: {target_z_axis}")

        # Return all 7 joint positions (including locked joint7)
        return self.data.qpos[:7].copy()

    def generate_trajectory(self, plan):
        """Generate waypoints from Gemini plan"""
        waypoints = []

        # Start from current joint configuration for smooth trajectory
        # Save actual robot state
        current_qpos = self.data.qpos[:7].copy()

        # 1. Move to approach position above object
        approach = plan["grasp_approach"]
        q_approach = self.inverse_kinematics_simple(np.array(approach))
        waypoints.append({"q": q_approach, "gripper": 0.04, "duration": 2.0})

        # 2. Move down to grasp object (start IK from previous waypoint)
        self.data.qpos[:7] = q_approach
        grasp_pos = plan["object_position"]
        q_grasp = self.inverse_kinematics_simple(np.array(grasp_pos))
        waypoints.append({"q": q_grasp, "gripper": 0.04, "duration": 1.5})

        # 3. Close gripper
        waypoints.append({"q": q_grasp, "gripper": 0.0, "duration": 1.0})

        # 4. Lift object (start IK from grasp position)
        self.data.qpos[:7] = q_grasp
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.1]
        q_lift = self.inverse_kinematics_simple(np.array(lift_pos))
        waypoints.append({"q": q_lift, "gripper": 0.0, "duration": 1.5})

        # 5. Move to place approach (start IK from lift position)
        self.data.qpos[:7] = q_lift
        place_approach = plan["place_approach"]
        q_place_approach = self.inverse_kinematics_simple(np.array(place_approach))
        waypoints.append({"q": q_place_approach, "gripper": 0.0, "duration": 2.0})

        # 6. Move down to place (start IK from approach position)
        self.data.qpos[:7] = q_place_approach
        place_pos = plan["target_position"]
        q_place = self.inverse_kinematics_simple(np.array(place_pos))
        waypoints.append({"q": q_place, "gripper": 0.0, "duration": 1.5})

        # 7. Open gripper
        waypoints.append({"q": q_place, "gripper": 0.04, "duration": 1.0})

        # 8. Retreat (start IK from place position)
        self.data.qpos[:7] = q_place
        retreat_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
        q_retreat = self.inverse_kinematics_simple(np.array(retreat_pos))
        waypoints.append({"q": q_retreat, "gripper": 0.04, "duration": 1.5})

        # Restore actual robot state
        self.data.qpos[:7] = current_qpos

        return waypoints

    def check_success(self):
        """Check if object is in target zone"""
        # Get target zone position
        target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
        target_body_id = self.model.geom_bodyid[target_geom_id]
        target_pos = self.data.xpos[target_body_id][:2]  # x, y only

        # Check each cube
        for cube_name in ["red_cube", "blue_cube"]:
            try:
                cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
                cube_pos = self.data.xpos[cube_body_id][:2]

                distance = np.linalg.norm(cube_pos - target_pos)
                if distance < 0.1:  # Within target zone radius
                    print(f"✓ SUCCESS! {cube_name} is in the target zone!")
                    return True
            except:
                pass

        return False

    def analyze_failure(self):
        """Analyze why the task failed - returns dict with numerical metrics"""
        target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
        target_body_id = self.model.geom_bodyid[target_geom_id]
        target_pos = self.data.xpos[target_body_id]

        failure_data = {
            "red_cube": {"grasped": False, "moved_distance": 0.0, "placement_error": 0.0, "dropped": False},
            "blue_cube": {"grasped": False, "moved_distance": 0.0, "placement_error": 0.0, "dropped": False},
            "summary": ""
        }

        failure_reasons = []

        # Check cube positions
        for cube_name in ["red_cube", "blue_cube"]:
            try:
                cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
                cube_pos = self.data.xpos[cube_body_id]

                # Check if cube moved at all
                initial_pos = np.array([0.5, 0.2, 0.025]) if cube_name == "red_cube" else np.array([0.4, -0.15, 0.025])
                moved_distance = np.linalg.norm(cube_pos[:2] - initial_pos[:2])
                failure_data[cube_name]["moved_distance"] = float(moved_distance)

                target_dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
                failure_data[cube_name]["placement_error"] = float(target_dist)

                if cube_pos[2] < 0.02:
                    failure_data[cube_name]["dropped"] = True

                if moved_distance < 0.02:
                    failure_reasons.append(f"{cube_name} was not grasped (didn't move from initial position)")
                else:
                    failure_data[cube_name]["grasped"] = True
                    if target_dist > 0.1:
                        failure_reasons.append(f"{cube_name} was grasped but placed {target_dist:.2f}m away from target")

                    # Check if dropped (low height)
                    if cube_pos[2] < 0.02:
                        failure_reasons.append(f"{cube_name} may have been dropped during transport")
            except:
                pass

        if not failure_reasons:
            failure_data["summary"] = "Object(s) in wrong location - trajectory may need adjustment"
        else:
            failure_data["summary"] = "; ".join(failure_reasons)

        return failure_data

    def control_step(self, target_q, target_gripper, dt):
        """Execute one control step with PID"""
        # All 7 arm joints (joint7 will be locked at 0.0 by target_q from IK)
        for i in range(self.n_joints):
            error = target_q[i] - self.data.qpos[i]
            control = self.pids[i].compute(error, dt)
            self.data.ctrl[i] = control

        # Gripper (actuator 7 controls the tendon for both fingers)
        # The gripper uses a 0-255 control range, remap from 0-0.04 to 0-255
        gripper_control_value = (target_gripper / 0.04) * 255.0
        self.data.ctrl[7] = gripper_control_value

    def run_with_viewer(self):
        """Main control loop with live MuJoCo viewer"""
        print("Starting LLM-Guided Robot Control")
        print("=" * 60)
        print(f"Attempt {self.attempt + 1}/{self.max_attempts}")

        # Launch viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat = np.array([0.3, 0, 0.2])

            start_time = time.time()
            waypoint_start = start_time
            current_waypoint_idx = 0
            target_q = self.data.qpos[:self.n_joints].copy()
            target_gripper = 0.04

            while viewer.is_running() and self.attempt < self.max_attempts:
                step_start = time.time()

                # Phase 1: Capture and analyze scene
                if self.task_phase == "capture":
                    print("\n[CAPTURE] Taking overhead camera image...")
                    image = self.get_camera_image()

                    # Save image for debugging
                    Image.fromarray(image).save(f"debug_view_attempt_{self.attempt}.png")
                    print(f"Saved debug_view_attempt_{self.attempt}.png")

                    # Log initial object positions
                    for cube_name in ["red_cube", "blue_cube"]:
                        try:
                            cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
                            cube_pos = self.data.xpos[cube_body_id]
                            self.trajectory_logger.log_object_position(cube_name, cube_pos, is_final=False)
                        except:
                            pass

                    self.task_phase = "plan"

                # Phase 2: Get plan from Gemini
                elif self.task_phase == "plan":
                    print("\n[PLAN] Asking Gemini for pick-and-place plan...")
                    image = self.get_camera_image()
                    plan = self.gemini.analyze_scene(image, self.execution_frames, self.structured_state)

                    # Clear for next attempt
                    self.execution_frames = []
                    self.trajectory_logger.reset()

                    print(f"\nGemini Analysis: {plan['analysis']}")
                    print(f"Target: {plan['target_object']} cube")
                    print(f"Object at: {plan['object_position']}")
                    print(f"Target at: {plan['target_position']}")
                    print(f"Confidence: {plan['confidence']}")

                    # Generate trajectory
                    print("\n[TRAJECTORY] Generating waypoints...")
                    self.waypoints = self.generate_trajectory(plan)
                    print(f"Generated {len(self.waypoints)} waypoints")


                    current_waypoint_idx = 0
                    waypoint_start = time.time()
                    self.task_phase = "execute"

                # Phase 3: Execute trajectory
                elif self.task_phase == "execute":
                    if current_waypoint_idx < len(self.waypoints) and len(self.waypoints) > 0:
                        wp = self.waypoints[current_waypoint_idx]
                        target_q = wp["q"]
                        target_gripper = wp["gripper"]

                        # Check if waypoint is complete
                        elapsed = time.time() - waypoint_start
                        if elapsed >= wp["duration"]:
                            # Log this waypoint execution (planned vs actual)
                            actual_ee_pos = self.data.xpos[self.ee_body_id].copy()
                            # Get planned position from IK (approximate from target_q)
                            planned_ee_pos = self.data.xpos[self.ee_body_id].copy()  # Use actual as approximation
                            waypoint_data = {
                                "waypoint": current_waypoint_idx,
                                "planned": list(planned_ee_pos),
                                "actual": list(actual_ee_pos),
                                "gripper": float(target_gripper),
                                "deviation": float(np.linalg.norm(planned_ee_pos - actual_ee_pos))
                            }
                            self.trajectory_logger.log_waypoint(
                                current_waypoint_idx,
                                planned_ee_pos,
                                actual_ee_pos,
                                target_gripper
                            )

                            # Capture only 3 key frames: initial (0), grasp (1), final (7)
                            if current_waypoint_idx in [0, 1, 7]:
                                self.execution_frames.append(self.get_camera_image())

                            # Debug: Print gripper orientation at waypoint completion
                            ee_mat = self.data.xmat[self.ee_body_id].reshape(3, 3)
                            z_axis = ee_mat[:, 2]
                            print(f"[EXECUTE] Completed waypoint {current_waypoint_idx}/{len(self.waypoints)} - Z-axis: [{z_axis[0]:.2f}, {z_axis[1]:.2f}, {z_axis[2]:.2f}], Joint7: {self.data.qpos[6]:.4f}")

                            # Real-time intervention: Check critical waypoints
                            if current_waypoint_idx in [2, 3]:  # After grasp (2) and lift (3)
                                print(f"[INTERVENTION] Asking Gemini to check waypoint {current_waypoint_idx}...")
                                current_image = self.get_camera_image()
                                object_states = self.get_object_states()

                                decision = self.gemini.check_waypoint(
                                    current_waypoint_idx,
                                    waypoint_data,
                                    current_image,
                                    object_states
                                )

                                print(f"[DECISION] {decision['decision'].upper()}: {decision['reason']}")

                                if decision['decision'] == 'abort':
                                    print(f"[ABORT] {decision['observed_issue']}")
                                    print(f"[CORRECTION] {decision['suggested_correction']}")

                                    # Store correction for next attempt
                                    self.last_correction = decision['suggested_correction']

                                    # Immediately reset and restart
                                    print(f"\n[RESET] Aborting trajectory and restarting immediately...")

                                    if self.attempt + 1 >= self.max_attempts:
                                        print("\n" + "=" * 60)
                                        print("MAX ATTEMPTS REACHED - TASK FAILED")
                                        print("=" * 60)
                                        return  # Exit run_with_viewer

                                    self.reset_simulation()
                                    self.attempt += 1
                                    self.task_phase = "capture"
                                    current_waypoint_idx = 0
                                    waypoint_start = time.time()

                                    print(f"\nStarting attempt {self.attempt + 1}/{self.max_attempts} (after early abort)")
                                    print(f"[STATUS] task_phase = {self.task_phase}")
                                    break  # Exit waypoint loop to restart

                            current_waypoint_idx += 1
                            waypoint_start = time.time()
                    else:
                        # Trajectory complete
                        print("\n[EXECUTE] Trajectory complete!")
                        self.task_phase = "check"

                # Phase 4: Check success
                elif self.task_phase == "check":
                    print("\n[CHECK] Checking if task succeeded...")
                    time.sleep(1.0)  # Let objects settle

                    # Log final object positions
                    for cube_name in ["red_cube", "blue_cube"]:
                        try:
                            cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
                            cube_pos = self.data.xpos[cube_body_id]
                            self.trajectory_logger.log_object_position(cube_name, cube_pos, is_final=True)
                        except:
                            pass

                    if self.check_success():
                        print("\n" + "=" * 60)
                        print("TASK COMPLETED SUCCESSFULLY!")
                        print("=" * 60)
                        time.sleep(3.0)
                        break
                    else:
                        # Analyze failure and create structured state for Gemini
                        failure_data = self.analyze_failure()
                        self.structured_state = self.trajectory_logger.get_structured_state(failure_data)

                        print(f"\n✗ Attempt {self.attempt + 1} failed. Resetting...")
                        print(f"Failure: {failure_data['summary']}")
                        self.attempt += 1

                        if self.attempt < self.max_attempts:
                            self.reset_simulation()
                            self.task_phase = "capture"
                            waypoint_start = time.time()
                            print(f"\nStarting attempt {self.attempt + 1}/{self.max_attempts}")
                        else:
                            print("\n" + "=" * 60)
                            print("MAX ATTEMPTS REACHED - TASK FAILED")
                            print("=" * 60)
                            time.sleep(3.0)
                            break

                # Control and simulation step (only during execution)
                dt = self.model.opt.timestep
                if self.task_phase == "execute":
                    self.control_step(target_q, target_gripper, dt)
                mujoco.mj_step(self.model, self.data)

                viewer.sync()

                # Maintain real-time simulation speed
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please run: export GEMINI_API_KEY='your-api-key-here'")
        return

    controller = RobotController()
    controller.run_with_viewer()


if __name__ == "__main__":
    main()
