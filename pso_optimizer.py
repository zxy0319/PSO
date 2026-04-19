from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

from pso_encoding_decoding import PSOSimulator, Wafer, build_default_factory_config, generate_wafers, SimulationResult


@dataclass
class Particle:
    position: List[float]  # Action vector (2N dimensions)
    velocity: List[float]  # Velocity vector
    best_position: List[float] = field(default_factory=list)
    best_fitness: float = float('inf')
    current_fitness: float = float('inf')

    def __post_init__(self):
        if not self.best_position:
            self.best_position = self.position.copy()


@dataclass
class PSOConfig:
    num_particles: int = 30
    max_iterations: int = 100
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive coefficient
    c2: float = 1.5  # Social coefficient
    num_wafers: int = 5
    makespan_weight: float = 0.5  # Weight for makespan in fitness
    tec_weight: float = 0.5  # Weight for total energy cost in fitness


class PSOOptimizer:
    def __init__(self, config: PSOConfig, simulator: PSOSimulator, wafers: Optional[List[Wafer]] = None):
        self.config = config
        self.simulator = simulator
        self.particles: List[Particle] = []
        self.global_best_position: List[float] = []
        self.global_best_fitness: float = float('inf')
        self.global_best_result: Optional[SimulationResult] = None

        # Use an externally provided wafer set if available; otherwise generate a reproducible one.
        self.wafers = wafers if wafers is not None else generate_wafers(
            self.config.num_wafers,
            self.simulator.num_operations,
            seed=42,
        )

        # Initialize particles
        self._initialize_particles()

    def _initialize_particles(self) -> None:
        """Initialize particles with random positions and velocities."""
        dimension = 2 * self.config.num_wafers
        for _ in range(self.config.num_particles):
            position = [random.random() for _ in range(dimension)]
            velocity = [random.uniform(-0.1, 0.1) for _ in range(dimension)]
            particle = Particle(position=position, velocity=velocity)
            self.particles.append(particle)

    def _evaluate_fitness(self, position: List[float]) -> Tuple[float, SimulationResult]:
        """Evaluate fitness of a position using the simulator."""
        # 传入固定的 wafers 进行解码仿真
        result = self.simulator.decode(position, self.wafers)

        # Multi-objective fitness: weighted sum
        fitness = (self.config.makespan_weight * result.makespan +
                  self.config.tec_weight * result.total_energy_cost)

        return fitness, result

    def _update_particle(self, particle: Particle) -> None:
        """Update particle velocity and position."""
        dimension = len(particle.position)
        r1, r2 = random.random(), random.random()

        for i in range(dimension):
            # Update velocity
            cognitive = self.config.c1 * r1 * (particle.best_position[i] - particle.position[i])
            social = self.config.c2 * r2 * (self.global_best_position[i] - particle.position[i])
            particle.velocity[i] = (self.config.w * particle.velocity[i] +
                                   cognitive + social)

            # Update position with bounds [0, 1]
            particle.position[i] += particle.velocity[i]
            particle.position[i] = max(0.0, min(1.0, particle.position[i]))

    def _update_personal_best(self, particle: Particle) -> None:
        """Update particle's personal best if current fitness is better."""
        if particle.current_fitness < particle.best_fitness:
            particle.best_fitness = particle.current_fitness
            particle.best_position = particle.position.copy()

    def _update_global_best(self, particle: Particle) -> None:
        """Update global best if particle's best fitness is better."""
        if particle.best_fitness < self.global_best_fitness:
            self.global_best_fitness = particle.best_fitness
            self.global_best_position = particle.best_position.copy()
            # Store the corresponding simulation result
            _, result = self._evaluate_fitness(particle.best_position)
            self.global_best_result = result

    def optimize(self) -> Tuple[List[float], SimulationResult, List[float]]:
        """
        Run PSO optimization.

        Returns:
            Tuple of (best_position, best_result, fitness_history)
        """
        fitness_history = []

        # Initial evaluation
        for particle in self.particles:
            particle.current_fitness, _ = self._evaluate_fitness(particle.position)
            self._update_personal_best(particle)
            self._update_global_best(particle)

        fitness_history.append(self.global_best_fitness)

        # Main PSO loop
        for iteration in range(self.config.max_iterations):
            for particle in self.particles:
                self._update_particle(particle)
                particle.current_fitness, _ = self._evaluate_fitness(particle.position)
                self._update_personal_best(particle)
                self._update_global_best(particle)

            fitness_history.append(self.global_best_fitness)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.config.max_iterations}, "
                      f"Best Fitness: {self.global_best_fitness:.2f}")

        return self.global_best_position, self.global_best_result, fitness_history


def run_pso_comparison():
    """Run PSO optimization and display results."""
    print("=== PSO Optimization for Semiconductor Scheduling ===\n")

    # Configuration
    config = PSOConfig(
        num_particles=50,  # Increased particles
        max_iterations=100,  # More iterations
        num_wafers=5,
        makespan_weight=0.7,  # Favor makespan more
        tec_weight=0.3,
        w=0.8,  # Higher inertia for exploration
        c1=2.0,  # Higher cognitive
        c2=2.0   # Higher social
    )

    # Setup simulator
    simulator = PSOSimulator(build_default_factory_config(batch_capacity=4))

    # Generate a shared wafer set for fair comparison
    wafers = generate_wafers(config.num_wafers, simulator.num_operations, seed=42)

    # Run PSO with the shared wafers
    optimizer = PSOOptimizer(config, simulator, wafers=wafers)
    best_position, best_result, fitness_history = optimizer.optimize()

    print("\n=== Optimization Results ===")
    print(f"Best Makespan: {best_result.makespan:.2f} hours")
    print(f"Best Total Energy Cost: {best_result.total_energy_cost:.2f} CNY")
    print(f"Best Fitness: {optimizer.global_best_fitness:.2f}")
    print(f"Initial Fitness: {fitness_history[0]:.2f}")
    print(f"Fitness Improvement: {((fitness_history[0] - optimizer.global_best_fitness) / fitness_history[0] * 100):.1f}%")

    print("\n=== Best Action Vector (first 10 values) ===")
    print([f"{x:.3f}" for x in best_position[:10]])

    print("\n=== Fitness History (every 5 iterations) ===")
    for i in range(0, len(fitness_history), 5):
        print(f"Iter {i:2d}: {fitness_history[i]:.2f}")

    # =========================================================
    # 修复报错区：在这里打印最优结果，因为这里有 best_result 变量
    # =========================================================
    print("=== Optimized Machine Schedule ===")
    for record in sorted(best_result.machine_schedule, key=lambda item: (item["start"], item["machine"], item["operation"])):
        jobs = ",".join(record["jobs"])
        batch_label = "BATCH" if record["machine_type"] == "batch" else "STD "
        print(
            f'{batch_label} {record["machine"]} {record["operation"]} '
            f'[{record["start"]:.1f}, {record["end"]:.1f}] jobs=[{jobs}]'
        )

    return best_position, best_result, fitness_history


if __name__ == "__main__":
    # 为了保证结果可复现，你可以加一行固定随机种子
    random.seed(42)
    np.random.seed(42)
    run_pso_comparison()