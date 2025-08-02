# internvl_service.py

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

# 假设 gr00t 库已安装，或者将其 service.py 文件放在同一目录下
from .service import BaseInferenceClient, BaseInferenceServer


class InternVLInferenceServer(BaseInferenceServer):
    """
    一个简单的服务器，它暴露一个 `get_action` 端点。
    它接收一个 "policy" 对象，并调用该对象的 `get_action` 方法。
    """

    def __init__(self, policy: Any, host: str = "*", port: int = 5555, api_token: str = None):
        """
        初始化服务器。
        
        Args:
            policy: 一个具有 'get_action(obs_dict)' 方法的对象。
            host: 绑定的主机地址。
            port: 监听的端口。
            api_token: 用于认证的可选API令牌。
        """
        super().__init__(host, port, api_token)
        # 注册一个名为 "get_action" 的端点，它会调用 policy 对象的 get_action 方法
        self.register_endpoint("get_action", policy.get_action)

    # def run(self):
    #     print(f"🚀 InternVL Inference Server is running on http://{self.host}:{self.port}")
    #     print("Waiting for client connections...")
    #     super().run()


class InternVLInferenceClient(BaseInferenceClient):
    """
    用于与 InternVLInferenceServer 通信的客户端。
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        通过网络调用服务器的 'get_action' 端点。
        
        Args:
            observations: 发送给服务器的观测数据字典。
        
        Returns:
            从服务器返回的动作字典或数组。
        """
        print(f"📞 Calling 'get_action' on server {self.host}:{self.port}")
        return self.call_endpoint("get_action", observations)