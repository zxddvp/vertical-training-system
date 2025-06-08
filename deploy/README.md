该目录包含集群部署示例。默认 `docker-compose up` 会启动一个 head 节点和一个 worker 节点。
如需增加 worker 数量，可运行：
```bash
docker-compose up --scale worker=2
```
实际物理多机可参考 compose 配置修改地址。
