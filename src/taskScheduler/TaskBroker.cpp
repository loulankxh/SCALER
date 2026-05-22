#include "TaskBroker.hpp"
#include "../utils/Logger.hpp"

using scalegann::Logger;

void TaskBroker::addTask(uint32_t id, size_t n, uint32_t d, uint32_t deg) {
    std::lock_guard<std::mutex> lock(mtx);
    tasks[id] = std::make_shared<ShardTask>(id, n, d, deg);
}

void TaskBroker::addDependency(uint32_t parent_id, uint32_t child_id) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.count(parent_id) && tasks.count(child_id)) {
        tasks[child_id]->in_degree++;
        tasks[parent_id]->children.push_back(child_id);
    }
}

void TaskBroker::addGPU(int gpu_id) {
    std::lock_guard<std::mutex> lock(mtx);
    active_gpus.insert(gpu_id);
    Logger::info("GPU {} added to active pool.", gpu_id);
}

void TaskBroker::removeGPU(int gpu_id) {
    std::lock_guard<std::mutex> lock(mtx);
    active_gpus.erase(gpu_id);
    Logger::spot(gpu_id, "GPU removed (Spot Interruption).");

    for (auto const& [id, task] : tasks) {
        if (task->status == ShardTask::RUNNING && task->gpu_id == gpu_id) {
            Logger::task(id, "Re-queued (was on GPU {}).", gpu_id);
            task->status = ShardTask::READY;
            task->gpu_id = -1;
        }
    }
}

void TaskBroker::refreshReadyStatus() {
    for (auto const& [id, task] : tasks) {
        if (task->status == ShardTask::PENDING && task->in_degree == 0) {
            task->status = ShardTask::READY;
        }
    }
}

std::vector<uint32_t> TaskBroker::collectReadyTasks() {
    std::vector<uint32_t> ready_ids;
    for (auto const& [id, task] : tasks) {
        if (task->status == ShardTask::READY) {
            ready_ids.push_back(id);
        }
    }
    return ready_ids;
}

void TaskBroker::prioritizeTasks(std::vector<uint32_t>& task_ids) {
    std::sort(task_ids.begin(), task_ids.end(), [&](uint32_t a, uint32_t b) {
        return tasks[a]->work_weight > tasks[b]->work_weight;
    });
}

std::vector<uint32_t> TaskBroker::binPackTasks(const std::vector<uint32_t>& candidates, int gpu_id, size_t max_vram) {
    std::vector<uint32_t> block;
    size_t current_vram = 0;

    for (uint32_t id : candidates) {
        auto task = tasks[id];
        if (current_vram + task->vram_req <= max_vram) {
            task->status = ShardTask::RUNNING;
            task->gpu_id = gpu_id;
            block.push_back(id);
            current_vram += task->vram_req;
        }
    }
    return block;
}

std::vector<uint32_t> TaskBroker::Scheduler(int gpu_id, size_t capacity, float alpha) {
    std::lock_guard<std::mutex> lock(mtx);
    
    // 1. Availability Check
    if (active_gpus.find(gpu_id) == active_gpus.end()) return {};

    // 2. State Refresh
    refreshReadyStatus();

    // 3. Candidate Selection
    std::vector<uint32_t> candidates = collectReadyTasks();

    // 4. Prioritization
    prioritizeTasks(candidates);

    // 5. Allocation (Bin Packing)
    size_t max_vram = (size_t)(capacity * alpha);
    return binPackTasks(candidates, gpu_id, max_vram);
}

void TaskBroker::updateStatus(uint32_t id, ShardTask::Status s) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.count(id)) {
        auto task = tasks[id];
        ShardTask::Status old_status = task->status;
        task->status = s;

        if (s == ShardTask::COMPLETED && old_status != ShardTask::COMPLETED) {
            for (uint32_t cid : task->children) {
                if (tasks[cid]->in_degree > 0) {
                    tasks[cid]->in_degree--;
                }
            }
        }
    }
}

ShardTask::Status TaskBroker::getTaskStatus(uint32_t id) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.count(id)) {
        return tasks[id]->status;
    }
    return ShardTask::FAILED;
}

float TaskBroker::getWorkWeight(uint32_t id) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.count(id)) {
        return tasks[id]->work_weight;
    }
    return 0.0f;
}

bool TaskBroker::isAllDone() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto const& [id, task] : tasks) {
        if (task->status != ShardTask::COMPLETED) return false;
    }
    return true;
}
