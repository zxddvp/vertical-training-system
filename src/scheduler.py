import ray
from typing import Any, Callable

def init_cluster(address: str = None) -> None:
    """初始化 Ray 集群。"""
    ray.init(address=address)

def distributed_map(func: Callable[[Any], Any], data: list) -> list:
    """在集群上并行执行 map 操作。"""
    remote_func = ray.remote(func)
    futures = [remote_func.remote(item) for item in data]
    return ray.get(futures)


def get_cluster_resources() -> dict:
    """返回当前 Ray 集群的资源信息。"""
    return ray.cluster_resources()
