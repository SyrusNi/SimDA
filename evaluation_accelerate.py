import os
import os
import random
import sys
import submitit
from omegaconf import DictConfig
#from trainer.accelerators.utils import nvidia_smi_gpu_memory_stats
import evaluation as classification
import argparse
from pathlib import Path
import uuid


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit", parents=[classification_parser])
    parser.add_argument("--name", default='eval', type=str, help="Name of submit")
    parser.add_argument("--mem_gb", default=68, type=float, help="Memory to request on each node")
    parser.add_argument("--ngpus", default=2, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--ntasks", default=2, type=int, help="Number of tasks to request on each node")
    parser.add_argument("--ncpus", default=1, type=int, help="Number of cpus to request for each task")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=720, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    #parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--partition", default="fvl", type=str, help="Partition where to submit")
    parser.add_argument("--qos", default="low", type=str, help="Quality of service")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument("--use_3090", action='store_true', help="3090!")
    #parser.add_argument('--comment', default="", type=str, help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/vhome").is_dir():
        p = Path(f"/vhome/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


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
        dist_env.master_port = rng.randint(10000, 20000) + dist_env.rank
        dist_env = dist_env.export()
        os.environ.update(**{
            "CUDA_LAUNCH_BLOCKING": "1",
            "NCCL_DEBUG": "info",
            "CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"],
        })
        #print(nvidia_smi_gpu_memory_stats())
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        print("Running training script")
        print(f"Local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}")
        num_processes = self.cfg.ntasks * self.cfg.nodes
        machine_rank = dist_env.rank // self.cfg.ntasks
        cmd = f"accelerate launch --dynamo_backend no --num_processes {num_processes} --num_machines {self.cfg.nodes} --use_deepspeed --machine_rank {machine_rank} --main_process_ip {dist_env.master_addr} --main_process_port {dist_env.master_port} evaluation.py --config={self.cfg.config}"
        print(f"Running command: {cmd}")
        print_env()
        os.system(cmd)

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


#@hydra.main(version_base=None, config_path="../conf", config_name="slurm_config")
def main() -> None:
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
    #print(cfg)
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.use_3090:
        kwargs['slurm_constraint'] = '3090'
    #if args.comment:
    #    kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=args.mem_gb,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ntasks,
        cpus_per_task=args.ncpus,
        nodes=args.nodes,
        timeout_min=args.timeout,  # max is 60 * 48 # first time 60 * 24 or less
        # Below are cluster dependent parameters
        slurm_qos = args.qos,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.name)

    task = Task(args)
    job = executor.submit(task)

    print("Submitted job_id:", job.job_id)
    #submitit.helpers.monitor_jobs([job], poll_frequency=60)


if __name__ == "__main__":
    main()