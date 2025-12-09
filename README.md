# continual-learning-neuroai

ref:
- Cheung, B., Terekhov, A., Chen, Y., Agrawal, P., & Olshausen, B. (2019, February 14). Superposition of many models into one. arXiv.org. https://arxiv.org/abs/1902.05522


replication results with slight modifications / variations to the original implementation and analysis:

### Performance of superposition model on sequentially learned tasks:
[![Superposition Performance](./plots/cl_performance_task_0.png)](./plots/cl_performance_task_0.png)

#### control for accuracy on current task (learning is continued):
[![Continued Learning Performance](./plots/cl_performance_task_0_control.png)](./plots/cl_performance_task_curr.png)

### representational shift analysis:
[![Representational Shift l0](./plots/representational_shift_layer_0.png)](./plots/representational_shift_layer_0.png)
[![Representational Shift l1](./plots/representational_shift_layer_1.png)](./plots/representational_shift_layer_1.png)
[![Representational Shift l2](./plots/representational_shift_layer_2.png)](./plots/representational_shift_layer_2.png)


### RDMs on later task over layers:
[![RSA RDMs](./plots/rdms_task_9.png)](./plots/rdms_task_9.png)
### RSA between context and baseline models:
[![RSA comparision](./plots/rsa_comparison_tasks_0_5_9.png)](./plots/rsa_comparison_tasks_0_5_9.png)