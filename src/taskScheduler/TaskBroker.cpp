#include "TaskBroker.hpp"
#include "../utils/Logger.hpp"
#include <chrono>

using scalegann::Logger;

void TaskBroker::addTask(uint32_t id, size_t n, uint32_t d, uint32_t deg) {
    std::lock_guard<std::mutex> lock(mtx);
    tasks[id] = std::make_shared<ShardTask>(id, n, d, deg);
}

void TaskBroker::addGPU(int gpu_id) {
    std::lock_guard<std::mutex> lock(mtx);
    active_gpus.insert(gpu_id);
    Logger::info("GPU {} added to active pool.", gpu_id);
}

void TaskBroker::removeGPU(int gpu_id) {
    std::lock_guard<std::mutex> lock(mtx);
    active_gpus.erase(gpu_id);
    Logger::spot(gpu_id, "GPU removed.");
}

void TaskBroker::refreshReadyStatus() {
    for (auto const& [id, task] : tasks) {
        if (task->status == ShardTask::PENDING) {
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

#include <random>

void TaskBroker::prioritizeTasks(std::vector<uint32_t>& task_ids) {
    if (random_mode) {
        static std::random_device rd;
        static std::mt19937 g(rd());
        std::shuffle(task_ids.begin(), task_ids.end(), g);
        return;
    }

    std::sort(task_ids.begin(), task_ids.end(), [&](uint32_t a, uint32_t b) {
        if (tasks[a]->retry_count != tasks[b]->retry_count) {
            return tasks[a]->retry_count > tasks[b]->retry_count;
        }
        return a < b; // Neutral tie-breaker (ID-based FIFO)
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
    
    if (active_gpus.find(gpu_id) == active_gpus.end()) return {};

    refreshReadyStatus();
    std::vector<uint32_t> candidates = collectReadyTasks();
    prioritizeTasks(candidates);

    size_t max_vram = (size_t)(capacity * alpha);
    auto block = binPackTasks(candidates, gpu_id, max_vram);
    
    if (!block.empty() && !has_started) {
        has_started = true;
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    return block;
}

void TaskBroker::updateStatus(uint32_t id, ShardTask::Status s) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.count(id)) {
        if (s == ShardTask::READY && tasks[id]->status == ShardTask::RUNNING) {
            tasks[id]->retry_count++;
            Logger::task(id, "Interrupted. Retry count: {}", tasks[id]->retry_count);
        }
        tasks[id]->status = s;
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

size_t TaskBroker::getTaskN(uint32_t id) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.count(id)) {
        return tasks[id]->n;
    }
    return 0;
}

bool TaskBroker::isAllDone() {
    std::lock_guard<std::mutex> lock(mtx);
    if (tasks.empty()) return false;

    for (auto const& [id, task] : tasks) {
        if (task->status != ShardTask::COMPLETED) return false;
    }

    if (has_started) {
        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        Logger::info("TaskBroker execution complete. Total time: {}ms", duration.count());
        has_started = false; 
    }
    return true;
}
