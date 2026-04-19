from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Wafer:
    wafer_id: str
    release_time: float = 0.0
    processing_times: List[float] = field(default_factory=list)  # 每个晶圆在每台机器上的加工时间
    next_op_index: int = 0
    ready_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def is_complete(self, num_operations: int) -> bool:
        return self.next_op_index >= num_operations

    def record_operation(self, op_name: str, machine_name: str, start: float, end: float) -> None:
        self.history.append(
            {
                "wafer_id": self.wafer_id,
                "operation": op_name,
                "machine": machine_name,
                "start": start,
                "end": end,
            }
        )


@dataclass
class MachineConfig:
    machine_id: str
    operation_index: int
    operation_name: str
    machine_type: str  # "standard" or "batch"
    batch_capacity: int
    processing_time: float
    power_consumption: float  # kW


@dataclass
class ScheduledBatch:
    machine_config: MachineConfig
    jobs: List[Wafer]
    start_time: float
    end_time: float


@dataclass
class SimulationResult:
    makespan: float
    schedule: List[Dict[str, Any]]
    wafer_histories: Dict[str, List[Dict[str, Any]]]
    total_energy_cost: float
    machine_schedule: List[Dict[str, Any]]


class PSOSimulator:
    def __init__(self, config: List[MachineConfig]):
        self.config = config
        self.num_operations = len(config)
        self.machines: List[MachineConfig] = config
        self.waiting_queues: List[List[Wafer]] = [[] for _ in range(self.num_operations)]
        self.machine_busy_until: List[float] = [0.0] * self.num_operations
        self.machine_jobs: List[List[Wafer]] = [[] for _ in range(self.num_operations)]
        self.event_queue: List[Tuple[float, int, str, int, Any]] = []
        self.event_counter: int = 0
        self.machine_histories: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_operations)]
        self.wafers: List[Wafer] = []
        self.batch_priorities: Dict[str, float] = {}
        self.preempt_priorities: Dict[str, float] = {}

    @staticmethod
    def split_action_vector(action_vector: List[float]) -> Tuple[List[float], List[float]]:
        n = len(action_vector) // 2
        if len(action_vector) != 2 * n:
            raise ValueError("Action vector length must be 2N.")
        return action_vector[:n], action_vector[n:]

    def build_priority_maps(self, action_vector: List[float], wafers: List[Wafer]) -> None:
        batch_scores, preempt_scores = self.split_action_vector(action_vector)
        if len(wafers) != len(batch_scores):
            raise ValueError("Number of wafers must equal N in action vector.")
        self.wafers = wafers
        self.batch_priorities = {
            wafer.wafer_id: batch_scores[idx] for idx, wafer in enumerate(wafers)
        }
        self.preempt_priorities = {
            wafer.wafer_id: preempt_scores[idx] for idx, wafer in enumerate(wafers)
        }

    def _push_event(self, event_time: float, event_type: str, index: int, payload: Any) -> None:
        heapq.heappush(self.event_queue, (event_time, self.event_counter, event_type, index, payload))
        self.event_counter += 1

    def _enqueue_wafer(self, wafer: Wafer, operation_index: int, event_time: float) -> None:
        wafer.ready_time = event_time
        self.waiting_queues[operation_index].append(wafer)
        self._push_event(event_time, "machine_check", operation_index, None)

    def _select_standard_job(self, queue: List[Wafer]) -> Wafer:
        return min(queue, key=lambda wafer: self.preempt_priorities[wafer.wafer_id])

    def _select_batch_jobs(self, queue: List[Wafer], capacity: int) -> List[Wafer]:
        sorted_queue = sorted(queue, key=lambda wafer: self.batch_priorities[wafer.wafer_id])
        return sorted_queue[:capacity]

    def _schedule_machine(self, machine_index: int, current_time: float) -> None:
        machine = self.machines[machine_index]
        if current_time < self.machine_busy_until[machine_index]:
            return
        queue = self.waiting_queues[machine_index]
        if not queue:
            return
        if machine.machine_type == "standard":
            selected = [self._select_standard_job(queue)]
            processing_time = selected[0].processing_times[machine_index]
        else:
            selected = self._select_batch_jobs(queue, machine.batch_capacity)
            # 批处理时间与所选wafers相关：取最大值（批处理需要等待最长的那个）
            processing_time = max(wafer.processing_times[machine_index] for wafer in selected)
        if not selected:
            return
        start_time = max(current_time, max(wafer.ready_time for wafer in selected))
        end_time = start_time + processing_time
        self.machine_busy_until[machine_index] = end_time
        self.machine_jobs[machine_index] = selected
        for wafer in selected:
            queue.remove(wafer)
            wafer.record_operation(machine.operation_name, machine.machine_id, start_time, end_time)
        self.machine_histories[machine_index].append({
            "machine": machine.machine_id,
            "operation": machine.operation_name,
            "machine_type": machine.machine_type,
            "jobs": [wafer.wafer_id for wafer in selected],
            "start": start_time,
            "end": end_time,
        })
        self._push_event(end_time, "machine_complete", machine_index, None)

    def _process_machine_complete(self, machine_index: int, event_time: float) -> None:
        completed = self.machine_jobs[machine_index]
        self.machine_jobs[machine_index] = []
        self.machine_busy_until[machine_index] = event_time
        for wafer in completed:
            wafer.next_op_index += 1
            if wafer.is_complete(self.num_operations):
                continue
            self._enqueue_wafer(wafer, wafer.next_op_index, event_time)
        self._push_event(event_time, "machine_check", machine_index, None)

    def decode(
        self,
        action_vector: List[float],
        wafers: List[Wafer],
        initial_time: float = 0.0,
    ) -> SimulationResult:
        self.build_priority_maps(action_vector, wafers)
        self.waiting_queues = [[] for _ in range(self.num_operations)]
        self.machine_busy_until = [0.0] * self.num_operations
        self.machine_jobs = [[] for _ in range(self.num_operations)]
        self.machine_histories = [[] for _ in range(self.num_operations)]
        self.event_queue = []
        self.wafers = wafers
        for wafer in wafers:
            wafer.next_op_index = 0
            wafer.ready_time = wafer.release_time
            wafer.history.clear()
            self._push_event(wafer.release_time, "wafer_ready", wafer.next_op_index, wafer)
        self._push_event(initial_time, "machine_check", -1, None)
        last_time = initial_time
        while self.event_queue:
            event_time, _, event_type, index, payload = heapq.heappop(self.event_queue)
            last_time = max(last_time, event_time)
            if event_type == "wafer_ready":
                wafer = payload
                self._enqueue_wafer(wafer, index, event_time)
            elif event_type == "machine_check":
                if index == -1:
                    for machine_index in range(self.num_operations):
                        self._schedule_machine(machine_index, event_time)
                else:
                    self._schedule_machine(index, event_time)
            elif event_type == "machine_complete":
                self._process_machine_complete(index, event_time)
        all_history = {wafer.wafer_id: wafer.history for wafer in wafers}
        makespan = max((entry["end"] for history in all_history.values() for entry in history), default=initial_time)
        schedule_list = [entry for history in all_history.values() for entry in history]
        machine_schedule = [entry for history in self.machine_histories for entry in history]
        total_energy_cost = calculate_energy_cost(self.machine_histories, self.config)
        return SimulationResult(
            makespan=makespan,
            schedule=schedule_list,
            wafer_histories=all_history,
            total_energy_cost=total_energy_cost,
            machine_schedule=machine_schedule,
        )


def get_tou_price(hour: float) -> float:
    """Get TOU price in CNY/kWh based on hour of day (0-24)."""
    hour_mod = hour % 24
    if 0 <= hour_mod < 6:
        return 0.4  # Valley
    elif 6 <= hour_mod < 8 or 12 <= hour_mod < 18 or 22 <= hour_mod < 24:
        return 0.8  # Shoulder
    elif 8 <= hour_mod < 12 or 18 <= hour_mod < 22:
        return 1.2  # Peak
    else:
        return 0.8  # Default to shoulder


def calculate_energy_cost(machine_histories: List[List[Dict[str, Any]]], machine_configs: List[MachineConfig]) -> float:
    """Calculate total energy cost from machine histories using TOU pricing."""
    total_cost = 0.0
    for machine_index, history in enumerate(machine_histories):
        config = machine_configs[machine_index]
        power = config.power_consumption
        for entry in history:
            start = entry["start"]
            end = entry["end"]
            # Integrate cost over time
            cost = 0.0
            t = start
            while t < end:
                hour_start = t
                next_hour = math.floor(t) + 1.0
                hour_end = min(end, next_hour)
                duration = hour_end - hour_start
                price = get_tou_price(hour_start)
                cost += price * power * duration
                t = hour_end
            total_cost += cost
    return total_cost


def build_default_factory_config(batch_capacity: int = 4) -> List[MachineConfig]:
    machine_types = [
        "standard",
        "batch",
        "standard",
        "batch",
        "standard",
        "batch",
        "standard",
        "batch",
        "standard",
    ]
    processing_times = [12.0, 28.0, 10.0, 24.0, 14.0, 26.0, 11.0, 22.0, 16.0]
    power_consumptions = [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0]  # kW
    return [
        MachineConfig(
            machine_id=f"M{k+1}",
            operation_index=k,
            operation_name=f"Op{k+1}",
            machine_type=machine_types[k],
            batch_capacity=batch_capacity if machine_types[k] == "batch" else 1,
            processing_time=processing_times[k],
            power_consumption=power_consumptions[k],
        )
        for k in range(9)
    ]


def generate_wafers(
    num_wafers: int,
    num_operations: int,
    seed: int = 42,
    release_range: Tuple[float, float] = (0.0, 10.0),
    process_range: Tuple[float, float] = (10.0, 30.0),
) -> List[Wafer]:
    """Generate wafers with diverse processing times and overlapped release times.
    
    Args:
        num_wafers: Number of wafers to generate
        num_operations: Number of operations per wafer
        seed: Random seed for reproducibility
        release_range: Range for wafer release times (default tight overlap)
        process_range: Range for processing times per operation
    """
    rng = random.Random(seed)
    wafers: List[Wafer] = []
    # 每个晶圆在不同工序上的处理时间不同，这样优化调度的效果才明显
    for i in range(num_wafers):
        release_time = rng.uniform(release_range[0], release_range[1])
        # 每个晶圆的处理时间不同，增加问题难度
        processing_times = [rng.uniform(process_range[0], process_range[1]) for _ in range(num_operations)]
        wafers.append(
            Wafer(
                wafer_id=f"W{i+1}",
                release_time=release_time,
                processing_times=processing_times,
            )
        )
    return wafers


if __name__ == "__main__":
    import random

    random.seed(42)  # 统一随机种子
    num_wafers = 5
    wafers = []
    for i in range(num_wafers):
        processing_times = [random.uniform(10, 50) for _ in range(9)]  # 9 operations
        wafers.append(Wafer(wafer_id=f"W{i+1}", release_time=0.0, processing_times=processing_times))
    action_vector = [random.random() for _ in range(2 * num_wafers)]
    simulator = PSOSimulator(build_default_factory_config(batch_capacity=4))
    result = simulator.decode(action_vector, wafers)

    print(f"makespan = {result.makespan:.1f}")
    print(f"total energy cost = {result.total_energy_cost:.2f} CNY")
    for record in sorted(result.schedule, key=lambda item: (item["end"], item["machine"], item["operation"])):
        print(
            f"{record['wafer_id']} -> {record['operation']} on {record['machine']} "
            f"[{record['start']:.1f}, {record['end']:.1f}]"
        )
