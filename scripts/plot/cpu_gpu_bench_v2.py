import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the MPI libraries benchmark.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--inputdir', type=str, default=os.path.join(project_dir, 'results'),
                    help='The directory with the results.')

parser.add_argument('-c', '--cases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-e', '--experiment', type=str, default="approx0-weak-scaling",
                    help='The directory with the results.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

args = parser.parse_args()
args.cases = args.cases.split(',')

device_results = {}
devices = ['cpu', 'cpu-gpu', 'gpu']
for c in args.cases:
    for d in devices:
        device_results[d] = []
        f = '{}/{}/{}/{}-{}/comm-comp-report.csv'.format(args.inputdir,d,c,args.experiment,d)
        print(f)
        print(os.path.exists(f))
        data = np.genfromtxt(f, delimiter='\t', dtype=str)
        for r in data:
            if (r[-6]=="total" or r[-6]=="total_time"):
                device_results[d].append((r[4],r[5],r[-5]))
                
        # print(data)
    for d in device_results:
        print(d)
        print(device_results[d])

    
    x = np.arange(3)  # the label locations
    width = 0.15  # the width of the bars
    cpu_times = np.array([float(x[2]) for x in device_results['cpu']])
    cpu_gpu_times = np.array([float(x[2]) for x in device_results['cpu-gpu']])
    gpu_times = np.array([float(x[2]) for x in device_results['gpu'][0::2]])
    gpu_2_times = np.array([float(x[2]) for x in device_results['gpu'][1::2]])

    ## normalizing
    cpu_gpu_times =  cpu_gpu_times / cpu_times
    gpu_times =  gpu_times / cpu_times
    gpu_2_times =  gpu_2_times / cpu_times
    cpu_times =  cpu_times / cpu_times

    print(gpu_2_times)
    fig, ax = plt.subplots(figsize=(15, 5))
    cpu = ax.bar(x - 3*width/2, cpu_times, width, label='cpu')
    cpu_gpu = ax.bar(x - width/2, cpu_gpu_times, width, label='cpu-gpu')
    gpu = ax.bar(x + width/2, gpu_times, width, label='1 GPU per node')
    gpu_2 = ax.bar(x + 3*width/2, gpu_2_times, width, label='2 GPUs per node')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized Time')
    ax.set_xlabel('Nodes')
    ax.set_title('{}_{}'.format(c,args.experiment))
    ax.set_xticks(x)
    ax.set_xticklabels([2**n for n in x])
    ax.legend()
    plt.show()

