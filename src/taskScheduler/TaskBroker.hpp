#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <algorithm>
#include <map>
#include <set>
#include <memory>
#include <chrono>


struct ShardTask {
    enum Status { PENDING, READY, RUNNING, COMPLETED, FAILED };

    uint32_t id;
    size_t n;
    size_t vram_req;
    float work_weight;
    
    Status status = PENDING;
    int gpu_id = -1;
    int retry_count = 0;

    ShardTask(uint32_t id, size_t n, uint32_t d, uint32_t deg) 
        : id(id), n(n) {
        vram_req = (n * d * 4) + (size_t)(n * deg * 6);
        work_weight = (float)n * d * deg;
    }
};

class TaskBroker {
public:
    TaskBroker() = default;

    void addTask(uint32_t id, size_t n, uint32_t d, uint32_t deg);

    std::vector<uint32_t> Scheduler(int gpu_id, size_t capacity, float alpha = 0.9f);

    void updateStatus(uint32_t id, ShardTask::Status s);

    ShardTask::Status getTaskStatus(uint32_t id);

    float getWorkWeight(uint32_t id);

    size_t getTaskN(uint32_t id);

    void removeGPU(int gpu_id);

    void addGPU(int gpu_id);

    bool isAllDone();

private:
    std::map<uint32_t, std::shared_ptr<ShardTask>> tasks;
    std::set<int> active_gpus;
    std::mutex mtx;

    // Performance tracking
    bool has_started = false;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

    void refreshReadyStatus();
    std::vector<uint32_t> collectReadyTasks();
    void prioritizeTasks(std::vector<uint32_t>& task_ids);
    std::vector<uint32_t> binPackTasks(const std::vector<uint32_t>& candidates, int gpu_id, size_t max_vram);
};
