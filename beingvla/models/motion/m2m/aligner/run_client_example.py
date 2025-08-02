# run_client_example.py

import numpy as np
import time

# 从我们创建的文件中导入客户端类
from internvl_service import InternVLInferenceClient

# 这是您想要的最终使用的 Policy 类
class Policy:
    """
    一个面向用户的策略类，通过网络与推理服务器交互。
    它管理动作块，以减少网络调用次数。
    """
    def __init__(self, host="localhost", port=5555, api_token=None, exec_chunk_size=10):
        print("Connecting to inference server...")
        # 注意：这里我们将我们自己的客户端改名为 ExternalRobotInferenceClient
        # 只是为了匹配您的示例代码，它实际上是 InternVLInferenceClient
        self.policy_client = InternVLInferenceClient(host=host, port=port, api_token=api_token)
        self.exec_chunk_size = exec_chunk_size # 这个参数现在由客户端管理
        self.reset()
        print("Connection successful.")

    def reset(self):
        """重置策略状态，清空动作缓存。"""
        self.t = 0
        self.action_chunk = []
        print("Policy has been reset.")

    def get_action(self, obs_dict: dict) -> np.ndarray:
        """
        获取单个时间步的动作。

        如果本地没有缓存的动作，它会向服务器请求一个新的动作块。
        否则，它会从缓存中返回下一个动作。

        Args:
            obs_dict (dict): 包含当前观测的字典。必须包含:
                - 'image' (np.ndarray): (H, W, 3) 格式的 RGB 图像。
                - 'state' (np.ndarray): 1D 的本体感受状态向量。
                - 'task_description' (str): 任务的文本描述。

        Returns:
            np.ndarray: 单个时间步的动作向量。
        """
        # 如果动作块为空，或者我们执行了一定数量的动作后（为了重新规划），
        # 就需要从服务器获取新的动作块。
        if self.t % self.exec_chunk_size == 0:
            print(f"Action chunk is empty or refresh interval reached (t={self.t}). Requesting new chunk...")
            start_time = time.time()
            
            # 调用客户端，通过网络发送观测数据
            new_action_chunk = self.policy_client.get_action(obs_dict)
            
            # 将 numpy 数组转换为列表，方便 pop 操作
            self.action_chunk = list(new_action_chunk)
            
            end_time = time.time()
            print(f"Received new action chunk of size {len(self.action_chunk)} in {end_time - start_time:.3f}s")

        # 从缓存的动作块中取出一个动作
        if not self.action_chunk:
             raise RuntimeError("Failed to get a valid action chunk from the server.")
             
        action = self.action_chunk.pop(0)
        self.t += 1
        
        return action


def main():
    # --- 客户端使用示例 ---
    print("--- Client Usage Example ---")

    # 1. 初始化策略客户端
    #    确保服务器已经在运行！
    #    exec_chunk_size=4 表示每4步向服务器请求一次新的16步规划
    try:
        policy = Policy(host="localhost", port=5555, exec_chunk_size=4)
    except Exception as e:
        print(f"Error: Could not connect to the server. Is it running? Details: {e}")
        return

    # 2. 准备一个模拟的观测字典 (obs_dict)
    #    这部分需要你根据你的机器人环境来填充真实数据
    #    - image: 必须是 (H, W, 3) 的 uint8 numpy 数组
    #    - state: 必须是 1D numpy 数组
    #    - task_description: 必须是字符串
    mock_obs_dict = {
        'image': np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        'state': np.random.rand(7).astype(np.float32), # 假设7自由度手臂
        'task_description': "pick up the red block"
    }
    print("\nPrepared a mock observation dictionary:")
    print(f"  - image shape: {mock_obs_dict['image'].shape}")
    print(f"  - state shape: {mock_obs_dict['state'].shape}")
    print(f"  - task: '{mock_obs_dict['task_description']}'")

    # 3. 模拟一个 rollout 循环
    print("\n--- Simulating a 10-step rollout ---")
    for step in range(10):
        print(f"\n[Step {step+1}]")
        action = policy.get_action(mock_obs_dict)
        
        print(f"-> Received action: {action}")
        print(f"   Action shape: {action.shape}, Dtype: {action.dtype}")

        # 在真实环境中，你会执行这个动作，然后获取下一个观测
        # 这里我们只是简单地循环
        time.sleep(0.1)


if __name__ == "__main__":
    main()