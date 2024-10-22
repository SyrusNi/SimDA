# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import evaluation as classification
import submitit


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


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        #import main as classification

        self._setup_gpu_args()
        num_processes = self.args.ngpus * self.args.nodes
        if num_processes > 1:
            cmd = f'accelerate launch --multi_gpu --num_processes={num_processes} --num_machines={self.args.nodes} evaluation.py --config={self.args.config}'
        else:
            cmd = f'accelerate launch evaluation.py --config={self.args.config}'
        print(f"Running command: {cmd}")
        print_env()
        os.system(cmd)
        print('amd yes')

    '''
    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)
    '''

    def _setup_gpu_args(self):
        #import submitit
        #from pathlib import Path
        print("Running task on slurm")
        print("exporting PyTorch distributed environment variables")
        os.environ.update(**{
            "CUDA_LAUNCH_BLOCKING": "1",
            "NCCL_DEBUG": "info",
            "CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"],
        })
        job_env = submitit.JobEnvironment()
        #self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        #self.args.gpu = job_env.local_rank
        #self.args.rank = job_env.global_rank
        #self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

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

    #args.dist_url = get_init_file().as_uri()
    #args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()