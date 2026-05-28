#include "TraceSimulator.hpp"
#include <iostream>

int main() {
    TaskBroker broker;

    // Create a larger set of tasks (e.g., 100 shards)
    for (uint32_t i = 0; i < 100; ++i) {
        // id, n, d, deg
        broker.addTask(i, 1000000, 128, 32); 
    }

    TraceSimulator simulator(broker, "tests/p3-trace.csv");
    simulator.run();

    return 0;
}
