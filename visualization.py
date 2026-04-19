import random
import matplotlib.pyplot as plt
from typing import List, Tuple
from pso_optimizer import PSOConfig
from pso_encoding_decoding import PSOSimulator, Wafer, build_default_factory_config, generate_wafers, SimulationResult
from dataclasses import dataclass, field

@dataclass
class Particle:
    position: List[float]
    velocity: List[float]
    best_position: List[float] = field(default_factory=list)
    best_fitness: Tuple[float, float] = (float('inf'), float('inf'))  # (makespan, cost)

class MultiObjectivePSO:
    def __init__(self, config: PSOConfig, simulator: PSOSimulator, wafers: List[Wafer]):
        self.config = config
        self.simulator = simulator
        self.wafers = wafers
        self.particles: List[Particle] = []
        self.archive: List[Tuple[List[float], SimulationResult]] = []  # Pareto archive
        self._initialize_particles()

    def _initialize_particles(self):
        dimension = 2 * self.config.num_wafers
        for _ in range(self.config.num_particles):
            position = [random.random() for _ in range(dimension)]
            velocity = [random.uniform(-0.1, 0.1) for _ in range(dimension)]
            particle = Particle(position=position, velocity=velocity)
            particle.best_position = position.copy()
            # Evaluate initial fitness
            fitness = self._evaluate_fitness(position)
            particle.best_fitness = fitness
            self.particles.append(particle)
            self._update_archive(position, fitness)

    def _evaluate_fitness(self, position: List[float]) -> Tuple[float, float]:
        result = self.simulator.decode(position, self.wafers)
        return result.makespan, result.total_energy_cost

    def dominates(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])

    def _update_archive(self, position: List[float], fitness: Tuple[float, float]):
        result = self.simulator.decode(position, self.wafers)

        # Update archive
        dominated = []
        for i, (pos, res) in enumerate(self.archive):
            if self.dominates(fitness, (res.makespan, res.total_energy_cost)):
                dominated.append(i)
            elif self.dominates((res.makespan, res.total_energy_cost), fitness):
                return

        for i in reversed(dominated):
            self.archive.pop(i)
        self.archive.append((position, result))

    def _select_guide(self, particle: Particle) -> List[float]:
        if self.archive:
            # Select random from archive
            return random.choice(self.archive)[0]
        else:
            # Random guide
            return [random.random() for _ in range(len(particle.position))]

    def optimize(self) -> List[Tuple[float, float]]:
        for iteration in range(self.config.max_iterations):
            for particle in self.particles:
                # Update velocity and position
                guide = self._select_guide(particle)
                r1, r2 = random.random(), random.random()

                for i in range(len(particle.position)):
                    cognitive = self.config.c1 * r1 * (particle.best_position[i] - particle.position[i])
                    social = self.config.c2 * r2 * (guide[i] - particle.position[i])
                    particle.velocity[i] = (self.config.w * particle.velocity[i] + cognitive + social)
                    particle.position[i] += particle.velocity[i]
                    particle.position[i] = max(0.0, min(1.0, particle.position[i]))

                # Mutation for exploration
                for i in range(len(particle.position)):
                    if random.random() < 0.01:  # 1% mutation rate per dimension
                        particle.position[i] += random.gauss(0, 0.05)
                        particle.position[i] = max(0.0, min(1.0, particle.position[i]))

                # Evaluate
                fitness = self._evaluate_fitness(particle.position)

                # Update personal best
                if self.dominates(fitness, particle.best_fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = fitness

                # Always try to update archive with current fitness
                self._update_archive(particle.position, fitness)

        return list(set([(res.makespan, res.total_energy_cost) for pos, res in self.archive]))

def random_search(simulator: PSOSimulator, wafers: List[Wafer], num_runs: int) -> List[Tuple[float, float]]:
    """Run random search and return results"""
    results = []
    for _ in range(num_runs):
        position = [random.random() for _ in range(2 * len(wafers))]
        result = simulator.decode(position, wafers)
        results.append((result.makespan, result.total_energy_cost))
    return results

def non_dominated_sort(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Perform non-dominated sorting on all points"""
    pareto_front = []
    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if i != j and dominates(p2, p1):
                dominated = True
                break
        if not dominated:
            pareto_front.append(p1)
    return list(set(pareto_front))  # 去重

def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])

def main():
    random.seed(42)
    config = build_default_factory_config(batch_capacity=4)
    simulator = PSOSimulator(config)
    num_wafers = 5

    # Generate fixed wafers for fair comparison
    wafers = generate_wafers(num_wafers, len(config), seed=42)

    # Random search
    print("Running random search...")
    random_results = random_search(simulator, wafers, 500)
    print(f"Random search completed: {len(random_results)} points")

    # MOPSO
    print("Running MOPSO...")
    pso_config = PSOConfig(num_particles=20, max_iterations=50, num_wafers=num_wafers, w=0.7, c1=2.0, c2=2.0)
    mopso = MultiObjectivePSO(pso_config, simulator, wafers)
    mopso_points = mopso.optimize()
    print(f"MOPSO completed: {len(mopso_points)} archive points")

    # Global Pareto front
    all_points = random_results + mopso_points
    global_pareto = non_dominated_sort(all_points)
    print(f"Global Pareto front: {len(global_pareto)} points")

    # Plot
    plt.figure(figsize=(10, 6))

    # Random search points
    random_makespans, random_costs = zip(*random_results)
    plt.scatter(random_makespans, random_costs, c='lightblue', alpha=0.6, s=20, label='Random Search')

    # MOPSO archive points
    if mopso_points:
        mopso_makespans, mopso_costs = zip(*mopso_points)
        plt.scatter(mopso_makespans, mopso_costs, c='red', marker='*', s=100, label='MOPSO Archive')

    # Global Pareto front
    if global_pareto:
        pareto_sorted = sorted(global_pareto)
        plt.plot([p[0] for p in pareto_sorted], [p[1] for p in pareto_sorted], 'r-', linewidth=2, label='Global Pareto Front')

    plt.xlabel('Makespan (minutes)')
    plt.ylabel('Total Energy Cost (CNY)')
    plt.title('Multi-Objective Optimization: Makespan vs Energy Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to optimization_results.png")

if __name__ == "__main__":
    main()