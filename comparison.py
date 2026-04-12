from __future__ import annotations

import random
import statistics
from typing import List, Tuple
from pso_encoding_decoding import PSOSimulator, Wafer, build_default_factory_config, SimulationResult
from pso_optimizer import PSOConfig, PSOOptimizer


def random_search(num_samples: int = 1000, num_wafers: int = 5) -> List[Tuple[List[float], SimulationResult]]:
    """Perform random search to establish baseline."""
    print(f"Performing random search with {num_samples} samples...")

    simulator = PSOSimulator(build_default_factory_config(batch_capacity=4))
    wafers = [Wafer(wafer_id=f"W{i+1}", release_time=0.0) for i in range(num_wafers)]
    dimension = 2 * num_wafers

    results = []
    for i in range(num_samples):
        action_vector = [random.random() for _ in range(dimension)]
        result = simulator.decode(action_vector, wafers)
        results.append((action_vector, result))

        if (i + 1) % 100 == 0:
            print(f"Random search: {i + 1}/{num_samples} completed")

    return results


def analyze_results(results: List[Tuple[List[float], SimulationResult]], method_name: str) -> None:
    """Analyze and print statistics of results."""
    makespans = [result.makespan for _, result in results]
    tecs = [result.total_energy_cost for _, result in results]

    print(f"\n=== {method_name} Results ===")
    print(f"Makespan - Min: {min(makespans):.2f}, Max: {max(makespans):.2f}, "
          f"Avg: {statistics.mean(makespans):.2f}, Std: {statistics.stdev(makespans) if len(makespans) > 1 else 0:.2f}")
    print(f"TEC - Min: {min(tecs):.2f}, Max: {max(tecs):.2f}, "
          f"Avg: {statistics.mean(tecs):.2f}, Std: {statistics.stdev(tecs) if len(tecs) > 1 else 0:.2f}")

    # Find Pareto front (non-dominated solutions)
    pareto_front = []
    for i, (action1, result1) in enumerate(results):
        dominated = False
        for j, (action2, result2) in enumerate(results):
            if i != j:
                if (result2.makespan <= result1.makespan and result2.total_energy_cost <= result1.total_energy_cost and
                    (result2.makespan < result1.makespan or result2.total_energy_cost < result1.total_energy_cost)):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append((action1, result1))

    print(f"Pareto front size: {len(pareto_front)}")

    # Show best solutions
    best_makespan = min(results, key=lambda x: x[1].makespan)
    best_tec = min(results, key=lambda x: x[1].total_energy_cost)

    print("\nBest Makespan Solution:")
    print(f"  Makespan: {best_makespan[1].makespan:.2f}")
    print(f"  TEC: {best_makespan[1].total_energy_cost:.2f}")

    print("\nBest TEC Solution:")
    print(f"  Makespan: {best_tec[1].makespan:.2f}")
    print(f"  TEC: {best_tec[1].total_energy_cost:.2f}")


def run_comparison():
    """Run comparison between random search and PSO."""
    print("=== Semiconductor Scheduling Optimization Comparison ===\n")

    # Random search
    random_results = random_search(num_samples=500, num_wafers=5)
    analyze_results(random_results, "Random Search")

    # PSO
    print("\nRunning PSO optimization...")
    config = PSOConfig(
        num_particles=50,
        max_iterations=100,
        num_wafers=5,
        makespan_weight=0.5,
        tec_weight=0.5,
        w=0.8,
        c1=2.0,
        c2=2.0
    )

    simulator = PSOSimulator(build_default_factory_config(batch_capacity=4))
    optimizer = PSOOptimizer(config, simulator)
    pso_position, pso_result, fitness_history = optimizer.optimize()

    pso_results = [(pso_position, pso_result)]
    analyze_results(pso_results, "PSO Optimization")

    # Compare with random search best
    best_random = min(random_results, key=lambda x: x[1].makespan + x[1].total_energy_cost)
    combined_fitness = best_random[1].makespan + best_random[1].total_energy_cost
    pso_fitness = pso_result.makespan + pso_result.total_energy_cost

    print("\n=== Comparison Summary ===")
    print(f"Random Search Best Combined Fitness: {combined_fitness:.2f}")
    print(f"PSO Best Combined Fitness: {pso_fitness:.2f}")
    print(f"PSO Improvement: {((combined_fitness - pso_fitness) / combined_fitness * 100):+.1f}%")


if __name__ == "__main__":
    run_comparison()