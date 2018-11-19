#!/usr/bin/env python

from __future__ import print_function

import multiprocessing
from multiprocessing import Pool
import os

######################################################################
## Hoa: 19.11.2018 Version: run_multi_processing.py
######################################################################
# This is a master file for postprocessing.py.
# The script starts multiple threads according to the number of
# CPU cores. Multiple instances of postprocessing.py are run parallel.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 19.11.2018 : first implemented
#
#
######################################################################

def run_process(process):
    try:
        os.system('python {}'.format(process))
    except Exception as e:
        print('Error in run_process: {}'.format(e))

def main():
    try:
        n_cpus = multiprocessing.cpu_count()
        print('Number of cpu cores: {}'.format(n_cpus))

        processes = ()

        for i in range(n_cpus-1):
            processes += 'postprocess.py',

        pool = Pool(processes = n_cpus-1)
        pool.map(run_process, processes)

    except Exception as e:
        print('Error in run_multi_postprocessing: {}'.format(e))


if __name__ == '__main__':
    main()
