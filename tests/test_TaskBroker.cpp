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

    // Dependencies: 1 -> 2
    broker.addDependency(1, 2);

    std::cout << "Initial state check..." << std::endl;
    assert(broker.isAllDone() == false);

    // GPU 0 requests tasks (Capacity 10MB)
    auto block0 = broker.Scheduler(0, 10000000);
    std::cout << "GPU 0 got " << block0.size() << " tasks." << std::endl;
    // Task 2 is NOT ready because 1 is not COMPLETED
    // Tasks 1 and 3 are READY. Priority 3 (1500*128*32) > 1 (1000*128*32)
    assert(block0.size() == 2);
    assert(block0[0] == 3); 
    assert(block0[1] == 1);

    // Simulate Spot Interruption on GPU 0
    std::cout << "Simulating Spot Interruption on GPU 0..." << std::endl;
    broker.removeGPU(0);
    
    assert(broker.getTaskStatus(1) == ShardTask::READY);
    assert(broker.getTaskStatus(3) == ShardTask::READY);

    // GPU 1 picks up tasks
    auto block1 = broker.Scheduler(1, 10000000);
    std::cout << "GPU 1 got " << block1.size() << " tasks after recovery." << std::endl;
    assert(block1.size() == 2);
    
    // Complete task 3
    broker.updateStatus(3, ShardTask::COMPLETED);
    assert(broker.getTaskStatus(3) == ShardTask::COMPLETED);

    // Complete task 1 -> Unlocks 2
    broker.updateStatus(1, ShardTask::COMPLETED);
    assert(broker.getTaskStatus(1) == ShardTask::COMPLETED);
    
    auto block2 = broker.Scheduler(1, 10000000);
    std::cout << "GPU 1 got " << block2.size() << " tasks after unlocking 2." << std::endl;
    assert(block2.size() == 1);
    assert(block2[0] == 2);

    broker.updateStatus(2, ShardTask::COMPLETED);
    assert(broker.isAllDone() == true);

    std::cout << "All TaskBroker tests passed!" << std::endl;
    return 0;
}
