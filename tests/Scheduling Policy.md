When constructing shard-wise indexes using cloud GPU instances, we require a task scheduler to assign appropriate index construction tasks to available GPU Instances. 

Our proposed solution involved designing a hybrid policy that prioritizes tasks that have suffered the most hardware preemptions. Among preempted tasks , the scheduler sorts by retry_count to ensure the most frequently interrupted tasks are rescued first.

Unattempted "fresh" tasks are assigned to available GPUs using randomized selection. By avoiding size-based policies we aimed to prevent large, high-risk shards from clustering together while being thrashed repeatedly.

To achieve this, the scheduler maintains the following information
1. A task list that tracks all pending index construction tasks.
2. A cloud instance list that keeps track of active GPU instances present in the pool. 

We derived this information from running the SIF1B dataset encapsulating the shard size and it's build time on the NVIDIA Tesla V100 GPUs with 16GB of HBM2 memory.  

The cloud instance list records the following statuses for each GPU instance: (1) Active: A GPU instance that has been successfully rented from cloud service provider’s instance pool is marked as active and added to the instance list. Conversely, if
the instance service is terminated, it is marked as inactive and removed from the list. (2) Available: Within the cloud instance list, if a GPU is currently executing a task, it is marked as unavailable; otherwise, it is available

The scheduler then uses the shard information and retry-count and assigns the task to one of the available GPU's from the pool. In case the GPU is disconnected mid-task, the retry-count for the task is incremented and the task is pushed back into the task queue for scheduler to re-prioritize and assign.

Figure 3 illustrates 