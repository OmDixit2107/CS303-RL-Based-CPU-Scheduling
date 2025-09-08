import random
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.priority = priority
        self.start_time = -1
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
    
    def __str__(self):
        return f"P{self.pid}(AT:{self.arrival_time}, BT:{self.burst_time})"

class CPUScheduler:
    def __init__(self):
        self.processes = []
        self.completed_processes = []
        self.current_time = 0
        self.timeline = []  # For Gantt chart
        
    def add_process(self, process):
        self.processes.append(process)
    
    def generate_random_processes(self, num_processes=5, max_arrival=10, max_burst=15):
        """Generate random processes for testing"""
        self.processes = []
        for i in range(num_processes):
            arrival = random.randint(0, max_arrival)
            burst = random.randint(1, max_burst)
            priority = random.randint(1, 5)
            self.add_process(Process(i+1, arrival, burst, priority))
        
        # Sort by arrival time
        self.processes.sort(key=lambda x: x.arrival_time)
        return self.processes
    
    def fcfs_scheduling(self):
        """First Come First Serve Scheduling"""
        print("=== FCFS Scheduling ===")
        self.reset_simulation()
        
        # Sort by arrival time
        processes = sorted(self.processes, key=lambda x: x.arrival_time)
        
        for process in processes:
            # Wait for process to arrive
            if self.current_time < process.arrival_time:
                self.current_time = process.arrival_time
            
            # Set start time
            if process.start_time == -1:
                process.start_time = self.current_time
            
            # Execute process
            self.timeline.append((process.pid, self.current_time, self.current_time + process.burst_time))
            self.current_time += process.burst_time
            
            # Calculate times
            process.completion_time = self.current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.burst_time
            
            self.completed_processes.append(process)
        
        return self.calculate_metrics()
    
    def sjf_scheduling(self):
        """Shortest Job First Scheduling (Non-preemptive)"""
        print("=== SJF Scheduling ===")
        self.reset_simulation()
        
        ready_queue = []
        remaining_processes = self.processes.copy()
        
        while remaining_processes or ready_queue:
            # Add arrived processes to ready queue
            arrived = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            for p in arrived:
                ready_queue.append(p)
                remaining_processes.remove(p)
            
            if ready_queue:
                # Select shortest job
                current_process = min(ready_queue, key=lambda x: x.burst_time)
                ready_queue.remove(current_process)
                
                if current_process.start_time == -1:
                    current_process.start_time = self.current_time
                
                # Execute process
                self.timeline.append((current_process.pid, self.current_time, 
                                   self.current_time + current_process.burst_time))
                self.current_time += current_process.burst_time
                
                # Calculate times
                current_process.completion_time = self.current_time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                
                self.completed_processes.append(current_process)
            else:
                # No process ready, advance time
                if remaining_processes:
                    self.current_time = min(p.arrival_time for p in remaining_processes)
        
        return self.calculate_metrics()
    
    def round_robin_scheduling(self, quantum=3):
        """Round Robin Scheduling"""
        print(f"=== Round Robin Scheduling (Quantum: {quantum}) ===")
        self.reset_simulation()
        
        ready_queue = deque()
        remaining_processes = self.processes.copy()
        
        # Reset remaining times
        for p in self.processes:
            p.remaining_time = p.burst_time
        
        while remaining_processes or ready_queue:
            # Add arrived processes to ready queue
            arrived = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            for p in arrived:
                ready_queue.append(p)
                remaining_processes.remove(p)
            
            if ready_queue:
                current_process = ready_queue.popleft()
                
                if current_process.start_time == -1:
                    current_process.start_time = self.current_time
                
                # Execute for quantum time or remaining time
                exec_time = min(quantum, current_process.remaining_time)
                
                self.timeline.append((current_process.pid, self.current_time, 
                                   self.current_time + exec_time))
                self.current_time += exec_time
                current_process.remaining_time -= exec_time
                
                # Check for newly arrived processes
                newly_arrived = [p for p in remaining_processes 
                               if p.arrival_time <= self.current_time]
                for p in newly_arrived:
                    ready_queue.append(p)
                    remaining_processes.remove(p)
                
                # If process not finished, add back to queue
                if current_process.remaining_time > 0:
                    ready_queue.append(current_process)
                else:
                    # Process completed
                    current_process.completion_time = self.current_time
                    current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                    current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                    self.completed_processes.append(current_process)
            else:
                # No process ready, advance time
                if remaining_processes:
                    self.current_time = min(p.arrival_time for p in remaining_processes)
        
        return self.calculate_metrics()
    
    def reset_simulation(self):
        """Reset simulation state"""
        self.completed_processes = []
        self.current_time = 0
        self.timeline = []
        
        # Reset process states
        for process in self.processes:
            process.start_time = -1
            process.completion_time = 0
            process.waiting_time = 0
            process.turnaround_time = 0
            process.remaining_time = process.burst_time
    
    def calculate_metrics(self):
        """Calculate and display performance metrics"""
        if not self.completed_processes:
            return {}
        
        avg_waiting = sum(p.waiting_time for p in self.completed_processes) / len(self.completed_processes)
        avg_turnaround = sum(p.turnaround_time for p in self.completed_processes) / len(self.completed_processes)
        total_time = max(p.completion_time for p in self.completed_processes)
        
        # CPU Utilization (assuming total time includes idle time)
        total_burst_time = sum(p.burst_time for p in self.processes)
        cpu_utilization = (total_burst_time / total_time) * 100 if total_time > 0 else 0
        
        metrics = {
            'avg_waiting_time': avg_waiting,
            'avg_turnaround_time': avg_turnaround,
            'total_time': total_time,
            'cpu_utilization': cpu_utilization
        }
        
        print(f"Average Waiting Time: {avg_waiting:.2f}")
        print(f"Average Turnaround Time: {avg_turnaround:.2f}")
        print(f"Total Execution Time: {total_time}")
        print(f"CPU Utilization: {cpu_utilization:.2f}%")
        print()
        
        return metrics
    
    def print_process_table(self):
        """Print process execution details"""
        print("\nProcess Execution Details:")
        print("PID\tAT\tBT\tST\tCT\tTAT\tWT")
        print("-" * 50)
        for p in self.completed_processes:
            print(f"{p.pid}\t{p.arrival_time}\t{p.burst_time}\t{p.start_time}\t"
                  f"{p.completion_time}\t{p.turnaround_time}\t{p.waiting_time}")
        print()
    
    def plot_gantt_chart(self, title="Gantt Chart"):
        """Create a Gantt chart visualization"""
        if not self.timeline:
            print("No timeline data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(item[0] for item in self.timeline))))
        color_map = {pid: colors[i] for i, pid in enumerate(set(item[0] for item in self.timeline))}
        
        for i, (pid, start, end) in enumerate(self.timeline):
            ax.barh(0, end - start, left=start, height=0.5, 
                   color=color_map[pid], alpha=0.8, 
                   edgecolor='black', linewidth=1)
            
            # Add process ID label
            ax.text(start + (end - start) / 2, 0, f'P{pid}', 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('CPU')
        ax.set_title(title)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, max(item[2] for item in self.timeline))
        
        # Remove y-axis ticks
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()

def run_comparison_demo():
    """Run a comparison of different scheduling algorithms"""
    print("CPU Scheduling Algorithm Comparison Demo")
    print("=" * 50)
    
    # Create scheduler and generate random processes
    scheduler = CPUScheduler()
    processes = scheduler.generate_random_processes(num_processes=5)
    
    print("Generated Processes:")
    for p in processes:
        print(f"Process {p.pid}: Arrival={p.arrival_time}, Burst={p.burst_time}")
    print()
    
    # Store results for comparison
    results = {}
    
    # Test FCFS
    scheduler.processes = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
    results['FCFS'] = scheduler.fcfs_scheduling()
    scheduler.print_process_table()
    scheduler.plot_gantt_chart("FCFS Scheduling")
    
    # Test SJF
    scheduler.processes = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
    results['SJF'] = scheduler.sjf_scheduling()
    scheduler.print_process_table()
    scheduler.plot_gantt_chart("SJF Scheduling")
    
    # Test Round Robin
    scheduler.processes = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
    results['RR'] = scheduler.round_robin_scheduling(quantum=3)
    scheduler.print_process_table()
    scheduler.plot_gantt_chart("Round Robin Scheduling (Q=3)")
    
    # Comparison chart
    algorithms = list(results.keys())
    avg_waiting = [results[alg]['avg_waiting_time'] for alg in algorithms]
    avg_turnaround = [results[alg]['avg_turnaround_time'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(algorithms, avg_waiting, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Average Waiting Time Comparison')
    ax1.set_ylabel('Time Units')
    
    ax2.bar(algorithms, avg_turnaround, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('Average Turnaround Time Comparison')
    ax2.set_ylabel('Time Units')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the demo
    run_comparison_demo()
    
    # Example of creating custom processes
    print("\n" + "="*50)
    print("Custom Process Example")
    print("="*50)
    
    custom_scheduler = CPUScheduler()
    
    # Add custom processes
    custom_scheduler.add_process(Process(1, 0, 8))
    custom_scheduler.add_process(Process(2, 1, 4))
    custom_scheduler.add_process(Process(3, 2, 9))
    custom_scheduler.add_process(Process(4, 3, 5))
    
    print("Custom Processes:")
    for p in custom_scheduler.processes:
        print(f"Process {p.pid}: Arrival={p.arrival_time}, Burst={p.burst_time}")
    
    # Test with custom processes
    custom_scheduler.round_robin_scheduling(quantum=2)
    custom_scheduler.print_process_table()
    custom_scheduler.plot_gantt_chart("Custom Round Robin (Q=2)")