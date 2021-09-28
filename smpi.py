import numpy
from mpi4py import MPI
from enum import Enum
import inspect
import logging


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


log = logging.getLogger(__name__ + ":" + str(rank))
log.setLevel(logging.DEBUG)
log.debug("module initialised")

class dist_type(Enum):
    local = 0
    broadcast = 1
    scatter = 2
    gather = 3

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class InRoot(metaclass=Singleton):
    def __init__(self):
        self.in_root_count = 0

    def __enter__(self):
        self.in_root_count += 1
        log.debug("enter: new root count {}".format(self.in_root_count))

    def __exit__(self, exc_type, exc_value, traceback):
        self.in_root_count -= 1
        log.debug("exit: new root count {}".format(self.in_root_count))

    def get_state(self):
        """
        @return True if in a root region, False otherwise
        """
        log.debug("get_state: {}".format(self.in_root_count))
        return self.in_root_count != 0

def _assert_not_in_root():
    if InRoot().get_state():
        raise RuntimeError("Can not call communicating functions while in smpi.root function, this will deadlock")


def root(func):
    def execute_root_only(*args):
        log.debug("exec root")
        with InRoot():
            if rank == 0:
                return func(*args)
            else:
                return None
    return execute_root_only

@root
def _calculate_distribution(data):
    elements = len(data)
    elem_per_rank = elements // size
    elem_left = elements - (elem_per_rank * size)
    count = []
    for _ in range(size):
        if elem_left > 0:
            count.append(elem_per_rank + 1)
            elem_left -= 1
        else:
            count.append(elem_per_rank)
    assert(sum(count) == elements)
    return count

@root
def _get_datatype(data):
    return data.dtype

@root
def _recv_container(count, dtype):
    return numpy.empty([count], dtype=dtype)

def _broadcast_data(data=None):
    '''
    broadcast data to all ranks
    '''
    _assert_not_in_root()
    recv = comm.bcast(data, root=0)
    return recv

def _scatter_data(data = None):
    '''
    scatterts a numpy 1D array along the available processes 
    '''
    _assert_not_in_root()
    if data is not None:
        if not isinstance(data, (numpy.ndarray)):
            raise TypeError("currently numpy arrays are supported")

    count = _calculate_distribution(data)
    local_count = comm.scatter(count)
    
    dtype = _get_datatype(data)
    dtype = comm.bcast(dtype)

    recv = numpy.empty([local_count], dtype=dtype)
    comm.Scatterv([data,count],recv)
    return recv

def _gather_data(data):
    '''
    gathers data into a numpy 1D array from the available processes 
    '''
    _assert_not_in_root()
    assert len(data) == data.size
    local_count = len(data)
    count = comm.reduce(local_count, op=MPI.SUM)
    
    dtype = _get_datatype(data)
    dtype = comm.bcast(dtype)

    recv = _recv_container(count, dtype)
    comm.Gatherv([data,local_count],recv)
    return recv


def distribute(*dist_types):
    '''
    decorates a function that has to be distributed,
    
    @distribute(broadcast, scatter, use_local)
    fun(A,B,C):
        pass

    paramteres with broadcast are copied to all ranks
    parameters with scatter are scattered among ranks
    parameters with local are not comunicated, but the local data is used.
    '''
    log.debug("call distribute")
    def new_func(func):
        sig = inspect.signature(func)
        assert len(dist_types) == len(sig.parameters)

        def distribute_data(*args):
            log.debug("call distribute_data")
            new_args = []
            for (i, (a, dt)) in enumerate(zip(args, dist_types)):
                try:
                    if dt == dist_type.broadcast:
                        a = _broadcast_data(a)
                    elif dt == dist_type.scatter:
                        a = _scatter_data(a)
                    elif dt == dist_type.local:
                        a = a
                    new_args.append(a)
                except TypeError as e:
                    raise TypeError("on parameter {}".format(i)) from e
            return func(*new_args)

        return distribute_data
    return new_func

def collect(*collect_types):
    '''
    decorates a function which resullts have to be collected,
    
    @collect(gather,broadcast,TODO)
    fun():
        return [A,B,C]

    parameters with gather are gethered from all ranks
    parameters with broadcast are broadcasted from root to all ranks
    '''
    log.debug("call collect")
    def new_func(func):
        def collect_data(*args, **kwds):
            log.debug("call collect_data")
            result = func(*args, **kwds)

            if result is None: # if @root has been used
                result = []
                for ct in collect_types:
                    assert ct == dist_type.broadcast
                    result.append(None)

            new_result = []
            for (i, (a, ct)) in enumerate(zip(result, collect_types)):
                try:
                    if ct == dist_type.gather:
                        a = _gather_data(a)
                    elif ct == dist_type.broadcast:
                        a = _broadcast_data(a)
                    new_result.append(a)
                except TypeError as e:
                    raise TypeError("on parameter {}".format(i)) from e
            return new_result
        return collect_data
    return new_func
