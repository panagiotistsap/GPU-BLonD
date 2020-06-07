import sys
import os
from mpi4py import MPI
import numpy as np
import logging
from functools import wraps
import socket

try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from ..utils import profile_mock as timing
    mpiprof = timing

from ..utils import bmath as bm

worker = None


def c_add_uint32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint32)
    y = np.frombuffer(ymem, dtype=np.uint32)
    bm.add(y, x, inplace=True)


add_op_uint32 = MPI.Op.Create(c_add_uint32, commute=True)


def c_add_uint16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint16)
    y = np.frombuffer(ymem, dtype=np.uint16)
    bm.add(y, x, inplace=True)


add_op_uint16 = MPI.Op.Create(c_add_uint16, commute=True)


def print_wrap(f):
    @wraps(f)
    def wrap(*args):
        msg = '[{}] '.format(worker.rank) + ' '.join([str(a) for a in args])
        if worker.isMaster:
            worker.logger.debug(msg)
            return f('[{}]'.format(worker.rank), *args)
        else:
            return worker.logger.debug(msg)
    return wrap


mpiprint = print_wrap(print)


class Worker:
    @timing.timeit(key='serial:init')
    @mpiprof.traceit(key='serial:init')
    def __init__(self):
        self.start_turn = 100
        self.start_interval = 500
        self.indices = {}
        self.interval = 500
        self.coefficients = {'particles': [0], 'times': [0.]}
        self.taskparallelism = False

        # Global inter-communicator
        self.intercomm = MPI.COMM_WORLD
        self.rank = self.intercomm.rank
        self.workers = self.intercomm.size

        # Setup TP intracomm
        self.hostname = MPI.Get_processor_name()
        self.hostip = socket.gethostbyname(self.hostname)

        # Create communicator with processes on the same host
        color = np.dot(np.array(self.hostip.split('.'), int)
                       [1:], [1, 256, 256**2])
        tempcomm = self.intercomm.Split(color, self.rank)
        temprank = tempcomm.rank
        # Break the hostcomm in neighboring pairs
        self.intracomm = tempcomm.Split(temprank//2, temprank)
        self.intraworkers = self.intracomm.size
        self.intrarank = self.intracomm.rank
        tempcomm.Free()
        self.log = False
        self.trace = False
        self.logger = MPILog(rank=self.rank)

    def initLog(self, log, logdir):
        self.log = log
        if not self.log:
            self.logger.disable()

    def initTrace(self, trace, tracefile):
        self.trace = trace
        if self.trace:
            mpiprof.mode = 'tracing'
            mpiprof.init(logfile=tracefile)

    def __del__(self):
        # if self.trace:
        mpiprof.finalize()

    @property
    def isMaster(self):
        return self.rank == 0

    @property
    def isFirst(self):
        return (self.intrarank == 0) or (self.taskparallelism is False)

    @property
    def isLast(self):
        return (self.intrarank == self.intraworkers-1) or (self.taskparallelism is False)

    # Define the begin and size numbers in order to split a variable of length size

    @timing.timeit(key='serial:split')
    @mpiprof.traceit(key='serial:split')
    def split(self, size):
        self.logger.debug('split')
        counts = [size // self.workers + 1 if i < size % self.workers
                  else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1])).astype(int)

        return displs[self.rank], counts[self.rank]

    # args are the buffers to fill with the gathered values
    # e.g. (comm, beam.dt, beam.dE)

    @timing.timeit(key='comm:gather')
    @mpiprof.traceit(key='comm:gather')
    def gather(self, var, size):
        self.logger.debug('gather')
        if self.isMaster:
            counts = np.empty(self.workers, int)
            sendbuf = np.array(len(var), int)
            self.intercomm.Gather(sendbuf, counts, root=0)
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, np.sum(counts))

            self.intercomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            sendbuf = np.array(len(var), int)
            self.intercomm.Gather(sendbuf, recvbuf, root=0)
            self.intercomm.Gatherv(var, recvbuf, root=0)
            return var

    @timing.timeit(key='comm:scatter')
    @mpiprof.traceit(key='comm:scatter')
    def scatter(self, var, size):
        self.logger.debug('scatter')
        if self.isMaster:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            # sendbuf = np.copy(var)
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            sendbuf = None
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv(sendbuf, recvbuf, root=0)

        return recvbuf

    @timing.timeit(key='comm:allreduce')
    @mpiprof.traceit(key='comm:allreduce')
    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32):
        self.logger.debug('allreduce')

        if dtype == np.uint32:
            op = add_op_uint32
        elif dtype == np.uint16:
            op = add_op_uint16
        else:
            print('Error: Not recognized dtype:{}'.format(dtype))
            exit(-1)

        if (recvbuf is None) or (sendbuf is recvbuf):
            self.intercomm.Allreduce(MPI.IN_PLACE, sendbuf, op=op)
        else:
            self.intercomm.Allreduce(sendbuf, recvbuf, op=op)

    @timing.timeit(key='serial:sync')
    @mpiprof.traceit(key='serial:sync')
    def sync(self):
        self.logger.debug('sync')
        self.intercomm.Barrier()

    @timing.timeit(key='serial:intraSync')
    @mpiprof.traceit(key='serial:intraSync')
    def intraSync(self):
        self.logger.debug('intraSync')
        self.intracomm.Barrier()

    @timing.timeit(key='serial:finalize')
    @mpiprof.traceit(key='serial:finalize')
    def finalize(self):
        self.logger.debug('finalize')
        if not self.isMaster:
            sys.exit(0)

    @timing.timeit(key='comm:sendrecv')
    @mpiprof.traceit(key='comm:sendrecv')
    def sendrecv(self, sendbuf, recvbuf):
        self.logger.debug('sendrecv')
        if self.isFirst and not self.isLast:
            self.intracomm.Sendrecv(sendbuf, dest=self.intraworkers-1, sendtag=0,
                                    recvbuf=recvbuf, source=self.intraworkers-1,
                                    recvtag=1)
        elif self.isLast and not self.isFirst:
            self.intracomm.Sendrecv(recvbuf, dest=0, sendtag=1,
                                    recvbuf=sendbuf, source=0, recvtag=0)

    @timing.timeit(key='comm:redistribute')
    @mpiprof.traceit(key='comm:redistribute')
    def redistribute(self, turn, beam, tcomp, tconst):
        self.coefficients['particles'].append(beam.n_macroparticles)
        self.coefficients['times'].append(tcomp)

        if len(self.coefficients['times']) < 2:
            # If it is the first time the function is called,
            # I need to do something different, I don't have enough data
            # to caluclate the coefficients.
            if self.intraworkers != 2:
                exit('Only support two workers per node for now!')
            # we exchange with the neighbour only the time it took us to compute
            recvbuf = np.empty(2 * self.intraworkers, dtype=float)
            self.intraworkers.Allgather(
                np.array([beam.n_macroparticles, tcomp]), recvbuf)
            req = None
            # let's say 1% of the particles
            P = np.sum(recvbuf[::2])
            size = int(1. * P/100)
            # There are only two values in the array, if more than the mean,
            # its the slow worker
            buf = np.empty(3*size, dtype=float)
            if tcomp > np.mean(recvbuf[1::2]):
                # if I am slower, I need to send
                i = beam.n_macroparticles - size
                buf[0:size] = beam.dE[i:i+size]
                buf[size:2*size] = beam.dt[i:i+size]
                buf[2*size:3*size] = beam.id[i:i+size]
                req = self.intraworkers.Isend(buf, 1-self.intrarank)
                # Then I need to resize local beam.dt and beam.dE, also
                # beam.n_macroparticles
                beam.dE = beam.dE[:beam.n_macroparticles-size]
                beam.dt = beam.dt[:beam.n_macroparticles-size]
                beam.id = beam.id[:beam.n_macroparticles-size]
                beam.n_macroparticles -= size
            else:
                req = self.intraworkers.Irecv(buf, 1-self.intrarank)
                req.Wait()
                # Then I need to resize local beam.dt and beam.dE, also
                # beam.n_macroparticles
                beam.dE = np.resize(beam.dE, beam.n_macroparticles + size)
                beam.dt = np.resize(beam.dt, beam.n_macroparticles + size)
                beam.id = np.resize(beam.id, beam.n_macroparticles + size)
                i = beam.n_macroparticles
                beam.dE[i:i+size] = buf[0:size]
                beam.dt[i:i+size] = buf[size:2*size]
                beam.id[i:i+size] = buf[2*size:3*size]
                beam.n_macroparticles += size
            return self.interval

        else:
            weights = np.ones(len(self.coefficients['times']))
            weights[-1] = np.sum(weights[:-1])
            p = np.polyfit(self.coefficients['particles'],
                           self.coefficients['times'], deg=1,
                           w=weights)
            latency = p[0]
            tconst += p[1]
            # latency = tcomp / beam.n_macroparticles
            recvbuf = np.empty(3 * self.workers, dtype=float)
            self.intercomm.Allgather(
                np.array([latency, tconst, beam.n_macroparticles]), recvbuf)

            latencies = recvbuf[::3]
            ctimes = recvbuf[1::3]
            Pi_old = recvbuf[2::3]

            P = np.sum(Pi_old)
            sum1 = np.sum(ctimes/latencies)
            sum2 = np.sum(1./latencies)
            Pi = (P + sum1 - ctimes * sum2)/(latencies * sum2)
            dPi = np.rint(Pi_old - Pi)

            for i in range(len(dPi)):
                if dPi[i] < 0 and -dPi[i] > Pi[i]:
                    dPi[i] = -Pi[i]
                elif dPi[i] > Pi[i]:
                    dPi[i] = Pi[i]

            transactions = calc_transactions(dPi, 2**4)[self.rank]
            if dPi[self.rank] > 0 and len(transactions) > 0:
                reqs = []
                tot_to_send = np.sum([t[1] for t in transactions])
                i = beam.n_macroparticles - tot_to_send
                for t in transactions:
                    # I need to send t[1] particles to t[0]
                    # buf[:t[1]] de, then dt, then id
                    buf = np.empty(3*t[1], dtype=float)
                    buf[0:t[1]] = beam.dE[i:i+t[1]]
                    buf[t[1]:2*t[1]] = beam.dt[i:i+t[1]]
                    buf[2*t[1]:3*t[1]] = beam.id[i:i+t[1]]
                    i += t[1]
                    # self.logger.critical(
                    #     '[{}]: Sending {} parts to {}.'.format(self.rank, t[1], t[0]))
                    reqs.append(self.intercomm.Isend(buf, t[0]))
                # Then I need to resize local beam.dt and beam.dE, also
                # beam.n_macroparticles
                beam.dE = beam.dE[:beam.n_macroparticles-tot_to_send]
                beam.dt = beam.dt[:beam.n_macroparticles-tot_to_send]
                beam.id = beam.id[:beam.n_macroparticles-tot_to_send]
                beam.n_macroparticles -= tot_to_send
                for req in reqs:
                    req.Wait()
                # req[0].Waitall(req)
            elif dPi[self.rank] < 0 and len(transactions) > 0:
                reqs = []
                recvbuf = []
                for t in transactions:
                    # I need to receive t[1] particles from t[0]
                    # The buffer contains: de, dt, id
                    buf = np.empty(3*t[1], float)
                    recvbuf.append(buf)
                    # self.logger.critical(
                    #     '[{}]: Receiving {} parts from {}.'.format(self.rank, t[1], t[0]))
                    reqs.append(self.intercomm.Irecv(buf, t[0]))
                for req in reqs:
                    req.Wait()
                # req[0].Waitall(req)
                # Then I need to resize local beam.dt and beam.dE, also
                # beam.n_macroparticles
                tot_to_recv = np.sum([t[1] for t in transactions])
                beam.dE = np.resize(
                    beam.dE, beam.n_macroparticles + tot_to_recv)
                beam.dt = np.resize(
                    beam.dt, beam.n_macroparticles + tot_to_recv)
                beam.id = np.resize(
                    beam.id, beam.n_macroparticles + tot_to_recv)
                i = beam.n_macroparticles
                for buf, t in zip(recvbuf, transactions):
                    beam.dE[i:i+t[1]] = buf[0:t[1]]
                    beam.dt[i:i+t[1]] = buf[t[1]:2*t[1]]
                    beam.id[i:i+t[1]] = buf[2*t[1]:3*t[1]]
                    i += t[1]
                beam.n_macroparticles += tot_to_recv

            if np.sum(np.abs(dPi))/2 < 1e-4 * P:
                self.interval = min(2*self.interval, 4000)
                return self.interval
            else:
                self.interval = self.start_interval
                return self.start_turn

    def report(self, turn, beam, tcomp, tcomm, tconst, tsync):
        latency = tcomp / beam.n_macroparticles
        self.logger.critical('[{}]: Turn {}, Tconst {:g}, Tcomp {:g}, Tcomm {:g}, Tsync {:g}, Latency {:g}, Particles {:g}'.format(
            self.rank, turn, tconst, tcomp, tcomm, tsync, latency, beam.n_macroparticles))

    def greet(self):
        self.logger.debug('greet')
        print('[{}]@{}: Hello World!'.format(self.rank, self.hostname))

    def print_version(self):
        self.logger.debug('version')
        # print('[{}] Library version: {}'.format(self.rank, MPI.Get_library_version()))
        # print('[{}] Version: {}'.format(self.rank,MPI.Get_version()))
        print('[{}] Library: {}'.format(self.rank, MPI.get_vendor()))

    def timer_start(self, phase):
        if phase not in self.times:
            self.times[phase] = {'start': MPI.Wtime(), 'total': 0.}
        else:
            self.times[phase]['start'] = MPI.Wtime()

    def timer_stop(self, phase):
        self.times[phase]['total'] += MPI.Wtime() - self.times[phase]['start']

    def timer_reset(self, phase):
        self.times[phase] = {'start': MPI.Wtime(), 'total': 0.}

    def initDLB(self, lb_type, lb_arg, n_iter):
        self.lb_turns = []
        self.lb_type = lb_type
        self.lb_arg = lb_arg
        if lb_type == 'times':
            if lb_arg != 0:
                intv = max(n_iter // (lb_arg+1), 1)
            else:
                intv = max(n_iter // (10 + 1), 1)
            self.lb_turns = np.arange(0, n_iter, intv)[1:]

        elif lb_type == 'interval':
            if lb_arg != 0:
                self.lb_turns = np.arange(0, n_iter, lb_arg)[1:]
            else:
                self.lb_turns = np.arange(0, n_iter, 1000)[1:]
        elif lb_type == 'dynamic':
            self.lb_turns = [self.start_turn]
        elif lb_type == 'reportonly':
            if lb_arg != 0:
                self.lb_turns = np.arange(0, n_iter, lb_arg)
            else:
                self.lb_turns = np.arange(0, n_iter, 100)
        self.dlb_times = {'tcomp': 0, 'tcomm': 0,
                          'tconst': 0, 'tsync': 0}
        return self.lb_turns

    def DLB(self, turn, beam):
        if turn not in self.lb_turns:
            return
        tcomp_new = timing.get(['comp:'])
        tcomm_new = timing.get(['comm:'])
        tconst_new = timing.get(['serial:'], exclude_lst=[
                                'serial:sync', 'serial:intraSync'])
        tsync_new = timing.get(['serial:sync', 'serial:intraSync'])
        if self.lb_type != 'reportonly':
            intv = self.redistribute(turn, beam,
                                     tcomp=tcomp_new-self.dlb_times['tcomp'],
                                     tconst=((tconst_new-self.dlb_times['tconst'])
                                             + (tcomm_new - self.dlb_times['tcomm'])))
        if self.lb_type == 'dynamic':
            self.lb_turns[0] += intv
        self.report(turn, beam, tcomp=tcomp_new-self.dlb_times['tcomp'],
                    tcomm=tcomm_new-self.dlb_times['tcomm'],
                    tconst=tconst_new-self.dlb_times['tconst'],
                    tsync=tsync_new-self.dlb_times['tsync'])
        self.dlb_times['tcomp'] = tcomp_new
        self.dlb_times['tcomm'] = tcomm_new
        self.dlb_times['tconst'] = tconst_new
        self.dlb_times['tsync'] = tsync_new


def calc_transactions(dpi, cutoff):
    trans = {}
    for i in range(len(dpi)):
        trans[i] = []
    arr = [{'val': i[1], 'id':i[0]} for i in enumerate(dpi)]

    # First pass is to prioritize transactions within the same node
    i = 0
    # e = len(arr)-1
    while i < len(arr)-1:
        if (arr[i]['val'] < 0) and (arr[i+1]['val'] > 0):
            s = i+1
            r = i
        elif (arr[i]['val'] > 0) and (arr[i+1]['val'] < 0):
            s = i
            r = i+1
        else:
            i += 2
            continue
        diff = int(min(abs(arr[s]['val']), abs(arr[r]['val'])))
        if diff > cutoff:
            trans[arr[s]['id']].append((arr[r]['id'], diff))
            trans[arr[r]['id']].append((arr[s]['id'], diff))
            arr[s]['val'] -= diff
            arr[r]['val'] += diff
        i += 2
    # Then the internode transactions
    arr = sorted(arr, key=lambda x: x['val'], reverse=True)
    s = 0
    e = len(arr)-1
    while (s < e) and (arr[s]['val'] >= 0) and (arr[e]['val'] <= 0):
        if arr[s]['val'] <= cutoff:
            s += 1
            continue
        if abs(arr[e]['val']) <= cutoff:
            e -= 1
            continue
        diff = int(min(abs(arr[s]['val']), abs(arr[e]['val'])))
        trans[arr[s]['id']].append((arr[e]['id'], diff))
        trans[arr[e]['id']].append((arr[s]['id'], diff))
        arr[s]['val'] -= diff
        arr[e]['val'] += diff

    return trans


class MPILog(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()
    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False
    """

    def __init__(self, rank=0, log_dir='./logs'):

        # Root logger on DEBUG level
        self.disabled = False
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.WARNING)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_name = log_dir+'/worker-%.3d.log' % rank
        # Console handler on INFO level
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        # console_handler.setFormatter(log_format)
        # self.root_logger.addHandler(console_handler)

        self.file_handler = logging.FileHandler(log_name, mode='w')
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(log_format)
        self.root_logger.addHandler(self.file_handler)
        logging.info("Initialized")
        # if debug == True:
        #     logging.debug("Logger in debug mode")

    def disable(self):
        """Disables all logging."""

        logging.info("Disable logging")
        # logging.disable(level=logging.NOTSET)
        # self.root_logger.setLevel(logging.NOTSET)
        # self.file_handler.setLevel(logging.NOTSET)
        self.root_logger.disabled = True
        self.disabled = True

    def debug(self, string):
        if self.disabled == False:
            logging.debug(string)

    def info(self, string):
        if self.disabled == False:
            logging.info(string)

    def critical(self, string):
        if self.disabled == False:
            logging.critical(string)


if worker is None:
    worker = Worker()
