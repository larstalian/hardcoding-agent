    def run_with_viewer(self):
        """Main control loop with live MuJoCo viewer - incremental execution"""
        print("Starting LLM-Guided Robot Control (Incremental Mode)")
        print("=" * 60)

        # Launch viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat = np.array([0.3, 0, 0.2])

            action_start = time.time()
            target_q = self.data.qpos[:self.n_joints].copy()
            target_gripper = 0.04
            current_action = None

            while viewer.is_running() and self.attempt < self.max_attempts:
                step_start = time.time()

                # Phase 1: Identify target object (only once per attempt)
                if self.task_phase == "identify_target":
                    print(f"\n[ATTEMPT {self.attempt + 1}/{self.max_attempts}] Identifying target object...")
                    image = self.get_camera_image()
                    Image.fromarray(image).save(f"debug_view_attempt_{self.attempt}.png")

                    # Use old analyze_scene just to pick which object
                    plan = self.gemini.analyze_scene(image)
                    self.target_object = plan['target_object'] + "_cube"

                    print(f"Target: {self.target_object}")
                    print(f"Target zone at: {self.target_zone_pos}")

                    self.task_phase = "execute_action"
                    action_start = time.time()

                # Phase 2: Execute one action at a time
                elif self.task_phase == "execute_action":
                    # Check if current action is complete
                    if current_action is None or (time.time() - action_start) >= 2.0:  # 2 sec per action
                        # Action complete, ask for next one
                        image = self.get_camera_image()
                        object_states = self.get_object_states()

                        # Record result of previous action
                        if current_action is not None:
                            ee_pos = self.data.xpos[self.ee_body_id].copy()
                            ee_mat = self.data.xmat[self.ee_body_id].reshape(3, 3)
                            z_axis = ee_mat[:, 2]

                            result = f"reached {ee_pos}, Z-axis=[{z_axis[0]:.2f},{z_axis[1]:.2f},{z_axis[2]:.2f}]"
                            current_action['result'] = result
                            self.trajectory_history.append(current_action)

                            print(f"\n[ACTION COMPLETE] {current_action['action']}: {result}")

                        # Ask Gemini for next action
                        print(f"\n[ASKING GEMINI] What's the next action? (History: {len(self.trajectory_history)} actions)")
                        next_action = self.gemini.ask_next_action(
                            image,
                            self.trajectory_history,
                            object_states,
                            self.target_object,
                            self.target_zone_pos
                        )

                        print(f"[NEXT ACTION] {next_action['action']}: {next_action['reasoning']}")

                        # Check if done or retry
                        if next_action['action'] == 'done':
                            print("\n[GEMINI] Task marked as done!")
                            self.task_phase = "check_success"
                        elif next_action['action'] == 'retry_previous':
                            print("\n[RETRY] Going back to retry previous action")
                            if len(self.trajectory_history) > 0:
                                self.trajectory_history.pop()  # Remove last action
                            # Will ask for next action again
                            current_action = None
                            action_start = time.time()
                        else:
                            # Compute IK for new target position
                            target_pos = np.array(next_action['target_pos'])
                            print(f"[IK] Computing for target: {target_pos}")
                            target_q = self.inverse_kinematics_simple(target_pos)
                            target_gripper = next_action['gripper']

                            current_action = next_action.copy()
                            current_action['target_q'] = target_q.tolist()
                            action_start = time.time()

                    # Execute current action
                    if current_action is not None:
                        dt = self.model.opt.timestep
                        self.control_step(target_q, target_gripper, dt)

                # Phase 3: Check success
                elif self.task_phase == "check_success":
                    print("\n[CHECK] Checking if task succeeded...")
                    time.sleep(1.0)  # Let objects settle

                    if self.check_success():
                        print("\n" + "=" * 60)
                        print("TASK COMPLETED SUCCESSFULLY!")
                        print("=" * 60)
                        time.sleep(3.0)
                        break
                    else:
                        print(f"\n✗ Attempt {self.attempt + 1} failed.")
                        self.attempt += 1

                        if self.attempt < self.max_attempts:
                            self.reset_simulation()
                            self.task_phase = "identify_target"
                            print(f"\nStarting attempt {self.attempt + 1}/{self.max_attempts}")
                        else:
                            print("\n" + "=" * 60)
                            print("MAX ATTEMPTS REACHED - TASK FAILED")
                            print("=" * 60)
                            time.sleep(3.0)
                            break

                # Simulation step
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # Maintain real-time simulation speed
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
