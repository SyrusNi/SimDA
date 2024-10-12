import os
import os
import random
import sys
import submitit
from omegaconf import DictConfig
from trainer.accelerators.utils import nvidia_smi_gpu_memory_stats


def print_env():
    for key in sorted(os.environ.keys()):
        if not (
                key.startswith(("SLURM_", "SUBMITIT_"))
                or key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        ):
            continue
        value = os.environ[key]
        print(f"{key}={value}")


class Task:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self):
        print("Running task on slurm")
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        rng = random.Random(dist_env._job_env.job_id)
        dist_env.master_port = rng.randint(10000, 20000)
        dist_env = dist_env.export()
        os.environ.update(**{
            "CUDA_LAUNCH_BLOCKING": "1",
            "NCCL_DEBUG": "info",
            "CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"],
        })
        print(nvidia_smi_gpu_memory_stats())
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        print("Running training script")
        print(f"Local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}")
        num_processes = self.cfg.slurm.n_processes * self.cfg.slurm.n_nodes
        machine_rank = dist_env.rank // self.cfg.slurm.n_processes
        cmd = f"accelerate launch --dynamo_backend no --num_processes {num_processes} --num_machines {self.cfg.slurm.n_nodes} --use_deepspeed --machine_rank {machine_rank} --main_process_ip {dist_env.master_addr} --main_process_port {dist_env.master_port} trainer/scripts/train.py {self.cfg.slurm.cmd}"
        print(f"Running command: {cmd}")
        print_env()
        if dist_env.local_rank == 0:
            os.system(cmd)
        else:
            print("Waiting for master to finish")

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


@hydra.main(version_base=None, config_path="../conf", config_name="slurm_config")
def main(cfg: DictConfig) -> None:
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=5900, stdoutToServer=True, stderrToServer=True)
    executor = submitit.AutoExecutor(folder="logs")
    print(cfg)
    slurm_kwargs = {
        "slurm_job_name": cfg.slurm.job_name,
        "slurm_partition": cfg.slurm.partition,
        "slurm_nodes": cfg.slurm.n_nodes,
        "slurm_additional_parameters": {
            "gpus": cfg.slurm.n_processes,
            "ntasks_per_node": cfg.slurm.n_processes,
        },
        "slurm_cpus_per_task": 12,
        "slurm_time": cfg.slurm.time_limit,
        "slurm_exclude": cfg.slurm.exclude if cfg.slurm.exclude else "",
        "stderr_to_stdout": True,
        "slurm_mem": "10GB",
    }
    executor.update_parameters(**slurm_kwargs)

    task = Task(cfg)
    job = executor.submit(task)
    submitit.helpers.monitor_jobs([job])


if __name__ == "__main__":
    sys.exit(main())