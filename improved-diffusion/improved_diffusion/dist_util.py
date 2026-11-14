# Licensed base: Diffusion-LM 759889d58ef38e2eed41a8c34db8032e072826f4 (Apache-2.0).
# OpenAI-derived portions retain their upstream MIT notices; see THIRD_PARTY_LICENSES.
# AP-MolDiff adapters: newly expressed from the frozen interface/behavior contract.
# Wsy-authorized local contributions remain subject to their MIT authorization.
# No assertion of full historical-model or paper reproduction is made.
import io
import os
import socket
import torch as th
import torch.distributed as dist
GPUS_PER_NODE = 1
SETUP_RETRY_COUNT = 3
MPI_AVAILABLE = False

def setup_dist(rank, world_size, port='12145'):
    if dist.is_initialized():
        return
    os.environ.update(MASTER_ADDR='localhost', MASTER_PORT=port)
    backend = 'gloo'
    if th.cuda.is_available():
        try:
            probe = th.zeros(1).cuda()
            del probe
            th.cuda.synchronize()
            backend = 'nccl'
        except Exception:
            backend = 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def dev():
    return th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def load_state_dict(path, **kwargs):
    import blobfile
    with blobfile.BlobFile(path, 'rb') as stream:
        contents = stream.read()
    return th.load(io.BytesIO(contents), **kwargs)

def sync_params(params):
    if dist.is_initialized():
        with th.no_grad():
            for tensor in params:
                dist.broadcast(tensor, src=0)

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind(('', 0))
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return connection.getsockname()[1]
