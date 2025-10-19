# LLM-Guided Robot Arm with MuJoCo and Gemini Vision

This project demonstrates a Franka Panda robot arm that uses Google Gemini's vision capabilities to perform pick-and-place tasks in MuJoCo simulation. The robot captures overhead camera views, sends them to Gemini for analysis, receives trajectory plans, and executes them using PID control.

## Features

- **Real Robot Model**: Uses the official Franka Emika Panda from MuJoCo Menagerie
- **Vision-Guided Control**: Gemini analyzes overhead camera views to plan grasps
- **PID Control**: Smooth trajectory following with PID controllers
- **Live Simulation**: Watch the robot execute tasks in real-time
- **Iterative Refinement**: Automatically retries with new plans on failure
- **Failure Recovery**: Resets and generates new trajectories until success

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Gemini API Key

Get your API key from https://makersuite.google.com/app/apikey

```bash
export GEMINI_API_KEY='your-api-key-here'
```

### 3. Run the Simulation

```bash
python llm_robot_control.py
```

## How It Works

### Architecture

1. **Capture Phase**: Overhead camera captures the workspace
2. **Planning Phase**: Image sent to Gemini for analysis
   - Gemini identifies object positions
   - Generates pick and place waypoints
   - Provides confidence estimate
3. **Execution Phase**: Robot follows trajectory using PID control
   - Approach object from above
   - Descend and close gripper
   - Lift and move to target zone
   - Open gripper and retreat
4. **Check Phase**: Verify if object reached target
   - Success: Task complete
   - Failure: Reset and try again with new Gemini plan

### PID Control

Each joint has an independent PID controller:
- **Kp = 10.0**: Proportional gain for responsiveness
- **Kd = 2.0**: Derivative gain for damping

### Inverse Kinematics

Uses MuJoCo's Jacobian-based iterative IK solver:
- Damped least squares for stability
- Converges to target end-effector positions
- Generates smooth joint trajectories

## Components

### Files

- `llm_robot_control.py`: Main control script
- `franka_panda/pick_place_scene.xml`: MuJoCo scene with objects
- `franka_panda/panda.xml`: Franka Panda robot model
- `requirements.txt`: Python dependencies

### Classes

- **PIDController**: Simple PID implementation for trajectory following
- **GeminiVisionController**: Interfaces with Gemini API for vision analysis
- **RobotController**: Main control loop integrating MuJoCo, PID, and Gemini

## Customization

### Adjust PID Gains

In `llm_robot_control.py`, modify:

```python
self.pids = [PIDController(kp=10.0, kd=2.0) for _ in range(self.n_joints)]
```

### Change Objects

Edit `franka_panda/pick_place_scene.xml` to add/modify objects:

```xml
<body name="your_object" pos="x y z">
  <freejoint/>
  <geom type="box" size="..." rgba="r g b a" mass="..."/>
</body>
```

### Modify Gemini Prompt

Update the prompt in `GeminiVisionController.analyze_scene()` to change how Gemini plans trajectories.

### Adjust Attempt Limit

```python
self.max_attempts = 5  # Change number of retry attempts
```

## Troubleshooting

### "GEMINI_API_KEY not set"
- Run: `export GEMINI_API_KEY='your-key'`

### Robot moves erratically
- Reduce PID gains (kp, kd)
- Increase waypoint durations
- Check IK convergence tolerance

### Gemini returns invalid JSON
- The system falls back to default waypoints
- Check your API key and quota
- Review the debug images saved as `debug_view_attempt_N.png`

### Simulation runs too fast/slow
- The viewer maintains real-time speed automatically
- Simulation timestep is set in the XML model

## Debug Output

The system saves camera images for each attempt:
- `debug_view_attempt_0.png`
- `debug_view_attempt_1.png`
- etc.

Review these to see what Gemini sees.

## Future Enhancements

- Obstacle avoidance
- Multi-object sequencing
- Force-based grasping
- Dynamic replanning during execution
- Fine-tuned vision prompts for better accuracy
- Integration with real Franka hardware

## License

This project uses:
- MuJoCo Menagerie models (Apache 2.0)
- Google Gemini API (Google's terms)
# hardcoding-agent
