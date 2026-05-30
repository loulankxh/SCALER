#include "TraceSimulator.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    TaskBroker broker;

    std::ifstream shard_file("tests/shard_info.csv");
    if (!shard_file.is_open()) {
        std::cerr << "Error: Could not open tests/shard_info.csv" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(shard_file, line); 
    while (std::getline(shard_file, line)) {
        std::stringstream ss(line);
        std::string id_str, points_str;
        if (std::getline(ss, id_str, ',') && std::getline(ss, points_str, ',')) {
            broker.addTask(std::stoul(id_str), std::stoul(points_str), 128, 32);
        }
    }

    TraceSimulator simulator(broker, "tests/p3-trace.csv", "tests/build_time.csv");
    simulator.run();

    return 0;
}
