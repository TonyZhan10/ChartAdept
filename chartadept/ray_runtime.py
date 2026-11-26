"""Launch torchrun on distinct Ray nodes and reserve VERL's per-GPU bundles."""
import asyncio
import os
import socket
import subprocess
import sys
import tempfile

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from chartadept.io import ROOT


def check_cluster(settings, work_dir):
    ray.init(address=settings['address'], log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node['Alive'] and node['Resources'].get('CPU', 0) > 0]
        eligible = [n for n in nodes if n['Resources'].get('GPU', 0) >= settings['gpus_per_node']]
        if len(eligible) < settings['nodes']:
            raise ValueError(f"Found {len(eligible)} nodes advertising {settings['gpus_per_node']} GPUs each; "
                             f"the selected topology needs {settings['nodes']}.")

        # Use each node's existing Ray interpreter to check the shared installation.
        @ray.remote(num_cpus=0)
        def missing_paths(paths):
            import os
            return [path for path in paths if not os.path.isfile(path)]

        with tempfile.NamedTemporaryFile(dir=work_dir) as marker:
            paths = [str(ROOT / 'chartadept/run.py'), sys.executable, marker.name]
            refs = [missing_paths.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node['NodeID'], soft=False)).remote(paths) for node in nodes]
            for node, missing in zip(nodes, ray.get(refs)):
                if missing:
                    raise RuntimeError(f"{node['NodeManagerAddress']} cannot see {missing}. "
                                       "Mount the package and work directory at the same paths on every Ray node.")
    finally:
        ray.shutdown()


def available_nodes(count, gpus, cpus):
    available = ray._private.state.available_resources_per_node()
    nodes = sorted(node['NodeManagerAddress'] for node in ray.nodes() if node['Alive']
                   and available.get(node['NodeID'], {}).get('GPU', 0) >= gpus
                   and available.get(node['NodeID'], {}).get('CPU', 0) >= cpus)
    if len(nodes) < count:
        raise RuntimeError(f'Need {count} distinct Ray nodes with {gpus} free GPU(s) and {cpus} free CPU(s) each; '
                           f'found {len(nodes)}. Run when the allocation is available.')
    return nodes[:count]


async def verify_broker(url):
    @ray.remote(num_cpus=0)
    def health(address):
        from urllib.request import ProxyHandler, build_opener
        opener = build_opener(ProxyHandler({}))
        with opener.open(address + '/health', timeout=10) as response:
            return response.status

    checks = [health.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node['NodeID'], soft=False)).remote(url) for node in ray.nodes() if node['Alive']]
    await asyncio.gather(*checks)


def reserve_verl_nodes(pool):
    """VERL expects one single-node placement group with one bundle per GPU."""
    gpus = pool.store[0]
    cpus = pool.max_colocate_count
    nodes = available_nodes(len(pool.store), gpus, gpus * cpus)
    groups = []
    try:
        for address in nodes:
            bundle = {'GPU': 1, 'CPU': cpus, f'node:{address}': .001}
            groups.append(placement_group([dict(bundle) for _ in range(gpus)], strategy='STRICT_PACK'))
        ray.get([group.ready() for group in groups], timeout=180)
    except BaseException:
        for group in groups:
            remove_placement_group(group)
        raise
    pool.pgs = groups
    print(f'VERL allocated {gpus} GPU(s) on each of {nodes}', flush=True)


class TorchNode:
    def __init__(self):
        self.process = None

    def rendezvous(self):
        with socket.socket() as listener:
            listener.bind(('', 0))
            return ray.util.get_node_ip_address(), listener.getsockname()[1]

    def execute(self, args, log):
        with open(log, 'w') as stream:
            print('Ray node', ray.util.get_node_ip_address(), 'devices', os.environ.get('CUDA_VISIBLE_DEVICES'),
                  file=stream, flush=True)
            self.process = subprocess.Popen([sys.executable, '-B', '-u', '-m', *args], cwd=ROOT,
                                            stdout=stream, stderr=subprocess.STDOUT)
            status = self.process.wait()
        if status:
            raise RuntimeError(f'Worker exited with status {status}. See {log}.')

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


def run_workers(settings, module_args, log, environment, distributed):
    nodes = settings['nodes'] if distributed else 1
    gpus = settings['gpus_per_node'] if distributed else 1
    cpus = max(2, gpus)
    runtime = {'py_executable': sys.executable, 'env_vars': environment}
    ray.init(address=settings['address'], runtime_env=runtime, log_to_driver=False)
    group, workers = None, []
    try:
        available_nodes(nodes, gpus, cpus)
        resources = {'GPU': gpus, 'CPU': cpus}
        group = placement_group([dict(resources) for _ in range(nodes)], strategy='STRICT_SPREAD')
        ray.get(group.ready(), timeout=180)
        actor = ray.remote(num_cpus=cpus, num_gpus=gpus, max_concurrency=2)(TorchNode)
        for index in range(nodes):
            placement = PlacementGroupSchedulingStrategy(placement_group=group, placement_group_bundle_index=index)
            workers.append(actor.options(scheduling_strategy=placement).remote())
        if distributed:
            master, port = ray.get(workers[0].rendezvous.remote())
        runs = []
        for rank, worker in enumerate(workers):
            args = module_args
            path = log
            if distributed:
                args = ['torch.distributed.run', f'--nnodes={nodes}', f'--nproc-per-node={gpus}',
                        f'--node-rank={rank}', f'--master-addr={master}', f'--master-port={port}', '-m', *module_args]
                path = log.with_suffix(f'.node-{rank}.log')
            runs.append(worker.execute.remote(args, str(path)))
        ray.get(runs)
    finally:
        try:
            if workers:
                ray.get([worker.stop.remote() for worker in workers], timeout=60)
        finally:
            for worker in workers:
                ray.kill(worker, no_restart=True)
            if group is not None:
                remove_placement_group(group)
            ray.shutdown()
