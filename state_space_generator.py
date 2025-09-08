import numpy as np
from cpu_scheduler_baseline import Process, CPUScheduler
import matplotlib.pyplot as plt
from collections import deque

class SchedulingStateSpace:
    """
    State Space Generator for RL-based CPU Scheduling
    
    This class defines and generates state representations that an RL agent
    can use to make scheduling decisions.
    """
    
    def __init__(self, max_processes=5, max_time_horizon=50):
        self.max_processes = max_processes
        self.max_time_horizon = max_time_horizon
        self.state_size = self._calculate_state_size()
        
    def _calculate_state_size(self):
        """Calculate the total size of state vector"""
        # Ready queue representation: max_processes * 3 (arrival, burst, remaining)
        ready_queue_size = self.max_processes * 3
        
        # Current time information: 1
        time_info_size = 1
        
        # System metrics: 3 (cpu_utilization, avg_waiting, queue_length)
        system_metrics_size = 3
        
        # Process priorities: max_processes
        priority_size = self.max_processes
        
        total_size = ready_queue_size + time_info_size + system_metrics_size + priority_size
        return total_size
    
    def extract_state(self, scheduler, current_time, ready_queue):
        """
        Extract state representation from current scheduling situation
        
        Args:
            scheduler: CPUScheduler instance
            current_time: Current simulation time
            ready_queue: List of processes currently ready to run
            
        Returns:
            numpy array representing the current state
        """
        state = np.zeros(self.state_size)
        idx = 0
        
        # 1. Ready Queue Information (normalized)
        ready_processes = ready_queue[:self.max_processes]  # Limit to max_processes
        
        for i in range(self.max_processes):
            if i < len(ready_processes) and ready_processes[i] is not None:
                process = ready_processes[i]
                
                # Normalize arrival time (0-1 based on max_time_horizon)
                state[idx] = min(process.arrival_time / self.max_time_horizon, 1.0)
                idx += 1
                
                # Normalize burst time (0-1 based on reasonable max burst time)
                state[idx] = min(process.burst_time / 20.0, 1.0)  # Assuming max burst = 20
                idx += 1
                
                # Normalize remaining time
                state[idx] = min(process.remaining_time / 20.0, 1.0)
                idx += 1
            else:
                # Empty slot - fill with zeros
                state[idx:idx+3] = [0.0, 0.0, 0.0]
                idx += 3
        
        # 2. Current Time Information (normalized)
        state[idx] = min(current_time / self.max_time_horizon, 1.0)
        idx += 1
        
        # 3. System Metrics
        # CPU Utilization (approximated)
        if scheduler.completed_processes:
            total_burst = sum(p.burst_time for p in scheduler.completed_processes)
            cpu_util = min(total_burst / max(current_time, 1), 1.0)
        else:
            cpu_util = 0.0
        state[idx] = cpu_util
        idx += 1
        
        # Average waiting time so far (normalized)
        if scheduler.completed_processes:
            avg_wait = sum(p.waiting_time for p in scheduler.completed_processes) / len(scheduler.completed_processes)
            state[idx] = min(avg_wait / 20.0, 1.0)  # Normalize by reasonable max wait time
        else:
            state[idx] = 0.0
        idx += 1
        
        # Ready queue length (normalized)
        state[idx] = len(ready_queue) / self.max_processes
        idx += 1
        
        # 4. Process Priorities (normalized)
        for i in range(self.max_processes):
            if i < len(ready_processes) and ready_processes[i] is not None:
                state[idx] = ready_processes[i].priority / 5.0  # Assuming max priority = 5
            else:
                state[idx] = 0.0
            idx += 1
        
        return state
    
    def get_action_mask(self, ready_queue):
        """
        Get valid actions (which processes can be scheduled)
        
        Returns:
            Binary mask indicating valid actions
        """
        action_mask = np.zeros(self.max_processes, dtype=bool)
        
        for i in range(min(len(ready_queue), self.max_processes)):
            if ready_queue[i] is not None:
                action_mask[i] = True
        
        return action_mask
    
    def visualize_state(self, state, title="Current State Representation"):
        """
        Visualize the state representation
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # 1. Ready Queue Processes
        ax1 = axes[0, 0]
        ready_queue_data = state[:self.max_processes * 3].reshape(self.max_processes, 3)
        
        x_pos = np.arange(self.max_processes)
        width = 0.25
        
        ax1.bar(x_pos - width, ready_queue_data[:, 0], width, label='Arrival Time', alpha=0.7)
        ax1.bar(x_pos, ready_queue_data[:, 1], width, label='Burst Time', alpha=0.7)
        ax1.bar(x_pos + width, ready_queue_data[:, 2], width, label='Remaining Time', alpha=0.7)
        
        ax1.set_xlabel('Process Slot')
        ax1.set_ylabel('Normalized Value')
        ax1.set_title('Ready Queue Process Information')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'P{i+1}' for i in range(self.max_processes)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. System Metrics
        ax2 = axes[0, 1]
        system_start = self.max_processes * 3 + 1  # Skip ready queue and time
        system_metrics = state[system_start:system_start + 3]
        metric_names = ['CPU Utilization', 'Avg Waiting Time', 'Queue Length']
        
        bars = ax2.bar(metric_names, system_metrics, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_ylabel('Normalized Value')
        ax2.set_title('System Performance Metrics')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, system_metrics):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3)
        
        # 3. Process Priorities
        ax3 = axes[1, 0]
        priority_start = self.max_processes * 3 + 1 + 3  # Skip ready queue, time, and system metrics
        priorities = state[priority_start:priority_start + self.max_processes]
        
        ax3.bar(range(self.max_processes), priorities, color='orange', alpha=0.7)
        ax3.set_xlabel('Process Slot')
        ax3.set_ylabel('Normalized Priority')
        ax3.set_title('Process Priorities')
        ax3.set_xticks(range(self.max_processes))
        ax3.set_xticklabels([f'P{i+1}' for i in range(self.max_processes)])
        ax3.grid(True, alpha=0.3)
        
        # 4. Complete State Vector
        ax4 = axes[1, 1]
        ax4.plot(range(len(state)), state, 'o-', linewidth=2, markersize=4)
        ax4.set_xlabel('State Element Index')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title(f'Complete State Vector (Size: {len(state)})')
        ax4.grid(True, alpha=0.3)
        
        # Add vertical lines to separate state components
        separators = [
            self.max_processes * 3,  # End of ready queue
            self.max_processes * 3 + 1,  # End of time info
            self.max_processes * 3 + 1 + 3,  # End of system metrics
        ]
        
        for sep in separators:
            ax4.axvline(x=sep, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def print_state_breakdown(self, state):
        """
        Print detailed breakdown of state components
        """
        print("=== STATE SPACE BREAKDOWN ===")
        print(f"Total State Size: {len(state)}")
        print()
        
        idx = 0
        
        # Ready Queue Information
        print("1. READY QUEUE PROCESSES:")
        for i in range(self.max_processes):
            arrival = state[idx] * self.max_time_horizon
            burst = state[idx + 1] * 20.0
            remaining = state[idx + 2] * 20.0
            
            print(f"   Process Slot {i+1}: Arrival={arrival:.1f}, Burst={burst:.1f}, Remaining={remaining:.1f}")
            idx += 3
        
        # Current Time
        print(f"\n2. CURRENT TIME: {state[idx] * self.max_time_horizon:.1f}")
        idx += 1
        
        # System Metrics
        print(f"\n3. SYSTEM METRICS:")
        print(f"   CPU Utilization: {state[idx]:.3f}")
        idx += 1
        print(f"   Average Waiting Time: {state[idx] * 20.0:.3f}")
        idx += 1
        print(f"   Queue Length Ratio: {state[idx]:.3f}")
        idx += 1
        
        # Priorities
        print(f"\n4. PROCESS PRIORITIES:")
        for i in range(self.max_processes):
            priority = state[idx] * 5.0
            print(f"   Process Slot {i+1}: Priority={priority:.1f}")
            idx += 1
        
        print("=" * 40)

class StateSpaceDemo:
    """
    Demonstration of state space generation during scheduling
    """
    
    def __init__(self):
        self.scheduler = CPUScheduler()
        self.state_generator = SchedulingStateSpace(max_processes=4, max_time_horizon=30)
    
    def simulate_with_state_tracking(self, processes):
        """
        Simulate scheduling while tracking state space evolution
        """
        print("=== STATE SPACE SIMULATION ===")
        
        # Reset scheduler
        self.scheduler.processes = processes
        self.scheduler.reset_simulation()
        
        # Initialize simulation variables
        current_time = 0
        ready_queue = deque()
        remaining_processes = processes.copy()
        states_history = []
        actions_history = []
        
        # Reset remaining times
        for p in processes:
            p.remaining_time = p.burst_time
        
        step = 0
        while remaining_processes or ready_queue:
            step += 1
            print(f"\n--- Simulation Step {step} (Time: {current_time}) ---")
            
            # Add newly arrived processes
            newly_arrived = [p for p in remaining_processes if p.arrival_time <= current_time]
            for p in newly_arrived:
                ready_queue.append(p)
                remaining_processes.remove(p)
                print(f"Process {p.pid} arrived (Burst: {p.burst_time})")
            
            # Convert ready_queue to list for state extraction
            ready_list = list(ready_queue)
            
            # Generate current state
            current_state = self.state_generator.extract_state(
                self.scheduler, current_time, ready_list
            )
            states_history.append(current_state.copy())
            
            # Get action mask (valid actions)
            action_mask = self.state_generator.get_action_mask(ready_list)
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            
            print(f"Ready Queue: {[f'P{p.pid}' for p in ready_list]}")
            print(f"Valid Actions: {valid_actions}")
            
            # Display state information
            self.state_generator.print_state_breakdown(current_state)
            
            if valid_actions:
                # For demo, use simple FCFS action (choose first available)
                action = valid_actions[0]
                selected_process = ready_list[action]
                actions_history.append(action)
                
                print(f"Action Taken: Schedule Process {selected_process.pid}")
                
                # Execute for 1 time unit (simplified)
                ready_queue.remove(selected_process)
                selected_process.remaining_time -= 1
                current_time += 1
                
                if selected_process.remaining_time > 0:
                    ready_queue.append(selected_process)  # Add back if not finished
                else:
                    print(f"Process {selected_process.pid} completed!")
                    selected_process.completion_time = current_time
                    selected_process.turnaround_time = selected_process.completion_time - selected_process.arrival_time
                    selected_process.waiting_time = selected_process.turnaround_time - selected_process.burst_time
                    self.scheduler.completed_processes.append(selected_process)
            else:
                # No processes ready, advance time
                if remaining_processes:
                    next_arrival = min(p.arrival_time for p in remaining_processes)
                    print(f"No processes ready. Advancing time to {next_arrival}")
                    current_time = next_arrival
                else:
                    break
            
            # Limit demo steps
            if step >= 10:
                print("\n[Demo limited to 10 steps]")
                break
        
        return states_history, actions_history
    
    def run_demo(self):
        """
        Run complete state space demonstration
        """
        # Create sample processes
        processes = [
            Process(1, 0, 4, priority=2),
            Process(2, 1, 2, priority=1),
            Process(3, 3, 3, priority=3),
            Process(4, 2, 1, priority=1)
        ]
        
        print("Demo Processes:")
        for p in processes:
            print(f"Process {p.pid}: Arrival={p.arrival_time}, Burst={p.burst_time}, Priority={p.priority}")
        
        # Run simulation with state tracking
        states, actions = self.simulate_with_state_tracking(processes)
        
        # Visualize some states
        if states:
            print(f"\nGenerated {len(states)} state representations")
            
            # Show first state
            self.state_generator.visualize_state(states[0], "Initial State")
            
            # Show middle state if available
            if len(states) > 3:
                self.state_generator.visualize_state(states[len(states)//2], "Mid-simulation State")
        
        return states, actions

def test_state_space_properties():
    """
    Test state space properties and characteristics
    """
    print("=== STATE SPACE PROPERTIES TEST ===")
    
    state_gen = SchedulingStateSpace(max_processes=3, max_time_horizon=20)
    
    print(f"State Space Size: {state_gen.state_size}")
    print(f"Max Processes: {state_gen.max_processes}")
    print(f"Max Time Horizon: {state_gen.max_time_horizon}")
    
    # Test with different scenarios
    test_cases = [
        # Empty ready queue
        [],
        # Single process
        [Process(1, 0, 5, priority=2)],
        # Multiple processes
        [Process(1, 0, 3, priority=1), Process(2, 1, 4, priority=3), Process(3, 2, 2, priority=2)]
    ]
    
    scheduler = CPUScheduler()
    
    for i, ready_queue in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {len(ready_queue)} processes ---")
        
        state = state_gen.extract_state(scheduler, current_time=5, ready_queue=ready_queue)
        action_mask = state_gen.get_action_mask(ready_queue)
        
        print(f"State shape: {state.shape}")
        print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
        print(f"Valid actions: {np.where(action_mask)[0].tolist()}")
        print(f"Non-zero elements: {np.count_nonzero(state)}/{len(state)}")

if __name__ == "__main__":
    # Run state space demonstration
    demo = StateSpaceDemo()
    states, actions = demo.run_demo()
    
    print("\n" + "="*60)
    
    # Test state space properties
    test_state_space_properties()
    
    print("\n" + "="*60)
    print("STATE SPACE GENERATION COMPLETE!")
    print("Ready for RL environment integration in Week 2!")