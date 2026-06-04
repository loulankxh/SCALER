#include "TraceSimulator.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct ShardInfo {
    uint32_t id;
    size_t n;
};

std::vector<ShardInfo> loadShards(const std::string& path) {
    std::vector<ShardInfo> shards;
    std::ifstream file(path);
    if (!file.is_open()) return shards;
    std::string line;
    std::getline(file, line); 
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string id_str, points_str;
        if (std::getline(ss, id_str, ',') && std::getline(ss, points_str, ',')) {
            shards.push_back({(uint32_t)std::stoul(id_str), (size_t)std::stoul(points_str)});
        }
    }
    return shards;
}

void run_policy_sim(const std::vector<ShardInfo>& shards) {
    TaskBroker broker;
    for (const auto& s : shards) {
        broker.addTask(s.id, s.n, 128, 32);
    }
    TraceSimulator sim(broker, "tests/p3-trace.csv", "tests/build_time.csv", 4, 2000, 0.1, SimulationMode::POLICY, "tests/sim_events_policy.csv");
    sim.run();
}

void run_random_sim(const std::vector<ShardInfo>& shards) {
    TaskBroker broker;
    broker.setRandomMode(true);
    for (const auto& s : shards) {
        broker.addTask(s.id, s.n, 128, 32);
    }
    TraceSimulator sim(broker, "tests/p3-trace.csv", "tests/build_time.csv", 4, 2000, 0.1, SimulationMode::RANDOM, "tests/sim_events_random.csv");
    sim.run();
}

void run_no_interruption_sim(const std::vector<ShardInfo>& shards) {
    TaskBroker broker;
    for (const auto& s : shards) {
        broker.addTask(s.id, s.n, 128, 32);
    }
    // No interruption mode: startup_delay = 0, uses loadStaticTrace
    TraceSimulator sim(broker, "tests/p3-trace.csv", "tests/build_time.csv", 4, 0, 0.1, SimulationMode::NO_INTERRUPTION, "tests/sim_events_nopreempt.csv");
    sim.run();
}

int main() {
    auto shards = loadShards("tests/shard_info.csv");
    if (shards.empty()) {
        std::cerr << "Error: No shards loaded." << std::endl;
        return 1;
    }

    std::cout << "=== Running Policy Simulation ===" << std::endl;
    run_policy_sim(shards);

    std::cout << "\n=== Running Random Simulation ===" << std::endl;
    run_random_sim(shards);

    std::cout << "\n=== Running No-Interruption Simulation ===" << std::endl;
    run_no_interruption_sim(shards);

    return 0;
}
