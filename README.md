# vertical-training-system

该项目旨在探索在消费级个人电脑组成的小规模集群上高效训练垂直分类模型。

## 项目结构
- `src/` Python 模块，实现数据处理、调度与模型训练。
- `data/` 示例数据集，真实数据请自行下载放入此目录。
- `deploy/` 部署脚本，包含 Dockerfile 与 docker-compose 配置。
- `tests/` 简单测试脚本，可验证训练流程。

## 快速开始
1. 安装依赖（非 Docker 环境）：
   ```bash
   pip install -r requirements.txt
   ```
2. 构建并启动集群：
   ```bash
   cd deploy
   docker-compose up --build
   ```
3. 运行测试：
   ```bash
   python tests/test_training.py
   ```
4. 运行示例脚本：
   ```bash
   python src/run_training.py
   ```

## 物理多机配置示例
docker-compose 文件展示了多节点的启动方式，可在不同机器上启动 Ray
并指定 `--address` 为 head 节点 IP，以实现真正的多机协作。
如需增加 worker 数量，可执行：
```bash
docker-compose up --scale worker=2
```
