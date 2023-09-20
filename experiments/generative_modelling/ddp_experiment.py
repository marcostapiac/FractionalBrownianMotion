import os

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    # If we run with torchrun, environment variables should be set by
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_process_group(backend)
    main2(backend)
    # Now destroy processes

def main2(backend):
    # At this point, as many processes as specified by the input arguments should have started
    print("Backend used is :: {}\n".format(backend))
    print("Worker with rank {}\n".format(int(os.environ["LOCAL_RANK"])))
    print("Number of threads for worker with rank {} is :: {}\n".format(int(os.environ["LOCAL_RANK"]),
                                                                      int(os.environ["OMP_NUM_THREADS"])))
    print("World size for rank {} is {}\n".format(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])))

if __name__ == "__main__":
    main()
    destroy_process_group()
