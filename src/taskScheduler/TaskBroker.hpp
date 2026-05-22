#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <algorithm>
#include <map>
#include <set>
#include <memory>


struct ShardTask {
    enum Status { PENDING, READY, RUNNING, COMPLETED, FAILED };

    uint32_t id;
    size_t vram_req;
    float work_weight;
    
    Status status = PENDING;
    int gpu_id = -1;
    
    // Dependency tracking
    uint32_t in_degree = 0; 
    std::vector<uint32_t> children;

    ShardTask(uint32_t id, size_t n, uint32_t d, uint32_t deg) 
        : id(id) {
        vram_req = (n * d * 4) + (size_t)(n * deg * 6);
        work_weight = (float)n * d * deg;
    }
};

class TaskBroker {
public:
    TaskBroker() = default;

    void addTask(uint32_t id, size_t n, uint32_t d, uint32_t deg);

    void addDependency(uint32_t parent_id, uint32_t child_id);

    std::vector<uint32_t> Scheduler(int gpu_id, size_t capacity, float alpha = 0.9f);

    void updateStatus(uint32_t id, ShardTask::Status s);

    ShardTask::Status getTaskStatus(uint32_t id);

    float getWorkWeight(uint32_t id);

    void removeGPU(int gpu_id);

    void addGPU(int gpu_id);

    bool isAllDone();

private:
    std::map<uint32_t, std::shared_ptr<ShardTask>> tasks;
    std::set<int> active_gpus;
    std::mutex mtx;
    void refreshReadyStatus();
    std::vector<uint32_t> collectReadyTasks();
    void prioritizeTasks(std::vector<uint32_t>& task_ids);
    std::vector<uint32_t> binPackTasks(const std::vector<uint32_t>& candidates, int gpu_id, size_t max_vram);
};
