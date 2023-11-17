import os

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP


@record
def main():
    # If we run with torchrun, environment variables should be set by
    print(torch.zeros(1).cuda())
    print(torch.__file__)
    print(torch.__path__)
    print("Torch version {}".format(torch.version.cuda))
    print("Is torch avaialble {}".format(torch.cuda.is_available()))
    print("Total device count available for GPUs is {}".format(torch.cuda.device_count()))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    print("Backend is ", backend)
    init_process_group(backend)
    if backend=="nccl":
        print(int(os.environ["LOCAL_RANK"]))
        print("Total number of available GPUs is {}\n".format(torch.cuda.device_count()))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return backend


def main2(backend, model):
    # At this point, as many processes as specified by the input arguments should have started
    print("Backend used is :: {}\n".format(backend))
    print("Worker with rank {}\n".format(int(os.environ["LOCAL_RANK"])))
    print("Number of threads for worker with rank {} is :: {}\n".format(int(os.environ["LOCAL_RANK"]),
                                                                        int(os.environ["OMP_NUM_THREADS"])))
    print("World size for rank {} is {}\n".format(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])))
    if backend=="nccl":
        print("Total number of available GPUs is {}\n".format(torch.cuda.device_count()))
        print("Current process device ID is {}\n".format(torch.cuda.get_device_name()))
        model = DDP(model.to(int(os.environ["LOCAL_RANK"])))

if __name__ == "__main__":
    backend = main()
    main2(backend=backend, model=torch.nn.Conv2d(16, 33, 3))
    destroy_process_group()
