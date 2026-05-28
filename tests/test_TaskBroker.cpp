#include "taskScheduler/TaskBroker.hpp"
#include <iostream>
#include <cassert>

int main() {
    TaskBroker broker;
    
    // Add GPUs
    broker.addGPU(0);
    broker.addGPU(1);

    // Add tasks
    // id, n, d, deg
    broker.addTask(1, 1000, 128, 32); // vram_req ~ 704000, work ~ 4096000
    broker.addTask(2, 2000, 128, 64); // Higher work
    broker.addTask(3, 1500, 128, 32); // Medium work

    std::cout << "Initial state check..." << std::endl;
    assert(broker.isAllDone() == false);

    // GPU 0 requests tasks (Capacity 10MB)
    auto block0 = broker.Scheduler(0, 10000000);
    std::cout << "GPU 0 got " << block0.size() << " tasks." << std::endl;
    // All tasks are READY. Priority 2 > 3 > 1 based on work weight
    assert(block0.size() == 3);
    assert(block0[0] == 2); 
    assert(block0[1] == 3);
    assert(block0[2] == 1);

    // Simulate Spot Interruption on GPU 0
    std::cout << "Simulating Spot Interruption on GPU 0..." << std::endl;
    broker.removeGPU(0);
    // Explicitly re-queue tasks that were "running" (simulated here)
    broker.updateStatus(2, ShardTask::READY);
    broker.updateStatus(3, ShardTask::READY);
    broker.updateStatus(1, ShardTask::READY);
    
    assert(broker.getTaskStatus(1) == ShardTask::READY);
    assert(broker.getTaskStatus(2) == ShardTask::READY);
    assert(broker.getTaskStatus(3) == ShardTask::READY);

    // GPU 1 picks up tasks
    auto block1 = broker.Scheduler(1, 10000000);
    std::cout << "GPU 1 got " << block1.size() << " tasks after recovery." << std::endl;
    // All 3 tasks should be picked up again
    assert(block1.size() == 3);
    // Because of retry_count, the order should remain the same (all were interrupted)
    assert(block1[0] == 2);
    assert(block1[1] == 3);
    assert(block1[2] == 1);
    
    // Complete tasks
    broker.updateStatus(3, ShardTask::COMPLETED);
    broker.updateStatus(1, ShardTask::COMPLETED);
    broker.updateStatus(2, ShardTask::COMPLETED);
    
    assert(broker.isAllDone() == true);

    std::cout << "All TaskBroker tests passed!" << std::endl;
    return 0;
}
