#!/usr/bin/env python

from __future__ import print_function

import multiprocessing
from multiprocessing import Pool
import os
import time

######################################################################
## Hoa: 19.11.2018 Version: run_multi_processing.py
######################################################################
# This is a master file for postprocessing.py.
# The script starts multiple threads according to the number of
# CPU cores. Multiple instances of postprocessing.py are run parallel.
# ----------------------------------------------------------------------
# Remarks: You must set the python's interpreters path to your individual
#          configuration !
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 19.11.2018 : first implemented
#
#
######################################################################

global interpreter_path

interpreter_path = r'C:\Users\ati\Anaconda3\envs\skycam\python.exe'

def run_process():
    try:
        os.system('{} postprocess.py'.format(interpreter_path))
    except Exception as e:
        print('Error in run_process: {}'.format(e))

def main():
    try:
        n_cpus = multiprocessing.cpu_count()
        print('Number of cpu cores: {}'.format(n_cpus))

        for i in range(n_cpus - 1):
            time.sleep(0.3)
            proc = multiprocessing.Process(target=run_process)
            proc.start()

        print('Done multiprocessing.')

    except Exception as e:
        print('Error in run_multi_postprocessing: {}'.format(e))


if __name__ == '__main__':
    main()
