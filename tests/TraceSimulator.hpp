#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <set>
#include "taskScheduler/TaskBroker.hpp"
#include "utils/Logger.hpp"

using scalegann::Logger;

struct TraceEvent {
    long timestamp;
    std::string action;
    int node_id;
};

struct RunningTask {
    uint32_t task_id;
    int gpu_id;
    long finish_time;
};

class TraceSimulator {
public:
    TraceSimulator(TaskBroker& broker, const std::string& trace_file) 
        : broker(broker), current_time(0), event_idx(0) {
        loadTrace(trace_file);
    }

    void run() {
        if (events.empty()) return;

        Logger::info("--- Starting Trace Simulation ---");

        while (!broker.isAllDone() || !running_pool.empty()) {
            processTraceEvents();
            checkTaskCompletions();
            scheduleNewTasks();

            current_time += 1000; 

            if (current_time > 100000000) { 
                Logger::error("Simulation timed out!");
                break;
            }
        }

        Logger::info("--- Simulation Finished ---");
        if (broker.isAllDone()) {
            Logger::success("All tasks completed.");
        }
    }

private:
    TaskBroker& broker;
    std::vector<TraceEvent> events;
    std::set<int> active_nodes;
    std::vector<RunningTask> running_pool;
    long current_time;
    size_t event_idx;

    void loadTrace(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string ts_str, action, node_str;
            if (std::getline(ss, ts_str, ',') && std::getline(ss, action, ',') && std::getline(ss, node_str, ',')) {
                TraceEvent e;
                e.timestamp = std::stol(ts_str);
                e.action = action;
                e.node_id = std::stoi(node_str.substr(4));
                events.push_back(e);
            }
        }
    }

    void processTraceEvents() {
        while (event_idx < events.size() && events[event_idx].timestamp <= current_time) {
            const auto& e = events[event_idx];
            if (e.action == "add") {
                broker.addGPU(e.node_id);
                active_nodes.insert(e.node_id);
            } else if (e.action == "remove") {
                broker.removeGPU(e.node_id);
                active_nodes.erase(e.node_id);
                handleInterruption(e.node_id);
            }
            event_idx++;
        }
    }

    void handleInterruption(int gpu_id) {
        auto it = running_pool.begin();
        while(it != running_pool.end()) {
            if (it->gpu_id == gpu_id) {
                Logger::spot(gpu_id, "Task {} interrupted.", it->task_id);
                broker.updateStatus(it->task_id, ShardTask::READY);
                it = running_pool.erase(it);
            } else {
                ++it;
            }
        }
    }

    void checkTaskCompletions() {
        auto it = running_pool.begin();
        while (it != running_pool.end()) {
            if (it->finish_time <= current_time) {
                Logger::task(it->task_id, "Finished on GPU {} at {}ms", it->gpu_id, current_time);
                broker.updateStatus(it->task_id, ShardTask::COMPLETED);
                it = running_pool.erase(it);
            } else {
                ++it;
            }
        }
    }

    void scheduleNewTasks() {
        for (int gpu_id : active_nodes) {
            if (isGPUBusy(gpu_id)) continue;

            auto block = broker.Scheduler(gpu_id, 16000000000ULL);
            for (uint32_t tid : block) {
                long duration = (long)(broker.getWorkWeight(tid) / 10000.0f); 

                running_pool.push_back({tid, gpu_id, current_time + duration});
                Logger::task(tid, "Started on GPU {} ({}ms)", gpu_id, duration);
            }
        }
    }

    bool isGPUBusy(int gpu_id) {
        for (const auto& rt : running_pool) {
            if (rt.gpu_id == gpu_id) return true;
        }
        return false;
    }
};
