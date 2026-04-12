from __future__ import annotations

import random
from pso_encoding_decoding import PSOSimulator, Wafer, build_default_factory_config


def test_different_wafer_counts():
    """Test the encoding-decoding system with different numbers of wafers."""
    print("=== Testing Different Wafer Counts ===\n")

    simulator = PSOSimulator(build_default_factory_config(batch_capacity=4))

    for num_wafers in [3, 5, 8, 10]:
        print(f"Testing with {num_wafers} wafers:")

        # Generate random action vector
        dimension = 2 * num_wafers
        action_vector = [random.random() for _ in range(dimension)]

        # Create wafers
        wafers = [Wafer(wafer_id=f"W{i+1}", release_time=0.0) for i in range(num_wafers)]

        # Decode
        result = simulator.decode(action_vector, wafers)

        print(f"  Makespan: {result.makespan:.2f} hours")
        print(f"  Total Energy Cost: {result.total_energy_cost:.2f} CNY")
        print(f"  Operations scheduled: {len(result.schedule)}")
        print()


if __name__ == "__main__":
    test_different_wafer_counts()