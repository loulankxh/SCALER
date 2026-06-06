#include "TraceSimulator.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

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

void run_policy_sim(const std::vector<ShardInfo>& shards, int iteration) {
    TaskBroker broker;
    for (const auto& s : shards) {
        broker.addTask(s.id, s.n, 128, 32);
    }
    std::string filename = "tests/policy_runs/sim_events_" + std::to_string(iteration) + ".csv";
    TraceSimulator sim(broker, "tests/p3-trace.csv", "tests/build_time.csv", 4, 2000, 0.1, SimulationMode::POLICY, filename);
    sim.run();
}

void run_random_sim(const std::vector<ShardInfo>& shards, int iteration) {
    TaskBroker broker;
    broker.setRandomMode(true); 
    for (const auto& s : shards) {
        broker.addTask(s.id, s.n, 128, 32);
    }
    std::string filename = "tests/random_runs/sim_events_" + std::to_string(iteration) + ".csv";
    TraceSimulator sim(broker, "tests/p3-trace.csv", "tests/build_time.csv", 4, 2000, 0.1, SimulationMode::RANDOM, filename);
    sim.run();
}

void run_no_interruption_sim(const std::vector<ShardInfo>& shards) {
    TaskBroker broker;
    for (const auto& s : shards) {
        broker.addTask(s.id, s.n, 128, 32);
    }
    TraceSimulator sim(broker, "tests/p3-trace.csv", "tests/build_time.csv", 4, 0, 0.1, SimulationMode::NO_INTERRUPTION, "tests/sim_events_nopreempt.csv");
    sim.run();
}

int main(int argc, char* argv[]) {
    int num_runs = 5; // Default
    if (argc > 1) {
        num_runs = std::stoi(argv[1]);
    }

    auto shards = loadShards("tests/shard_info.csv");
    if (shards.empty()) {
        std::cerr << "Error: No shards loaded." << std::endl;
        return 1;
    }

    std::cout << "=== Running " << num_runs << " Policy Simulations (Hybrid) ===" << std::endl;
    for(int i=0; i < num_runs; ++i) {
        if (i % 10 == 0) std::cout << "Policy Iteration " << i << "..." << std::endl;
        run_policy_sim(shards, i);
    }

    std::cout << "\n=== Running " << num_runs << " Randomized Baseline Simulations ===" << std::endl;
    for (int i = 0; i < num_runs; ++i) {
        if (i % 10 == 0) std::cout << "Random Iteration " << i << "..." << std::endl;
        run_random_sim(shards, i);
    }

    std::cout << "\n=== Running No-Interruption Simulation ===" << std::endl;
    run_no_interruption_sim(shards);

    return 0;
}
