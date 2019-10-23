import numpy as np
import pandas as pd
import copyreg, types
import multiprocessing as mp
import sys
import time
import datetime as dt
from tqdm import tqdm
import pickle


# -------------------------------------
# registry stuff for  serialization pickling/unpickling

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self # class instance object
    cls = method.im_class # class that asked for the method
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name] # from
        except KeyError: pass
        else: break
    return func.__get__(obj, cls) # get attribute (instance, owner)


copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# ---------------------------------
# parallel computing
# ---------------------------------


def expand_call(kargs):
    func = kargs['func']  # value: the function to be used (you unwrap fct from dictionary)
    del kargs['func'] # kill key-value pair (delete function from dictionary)
    out = func(**kargs) # execute function, this is where function execution happens, kargs includes fct + all params
    return out


def mp_pandas_obj(func, pd_obj, numThreads=8, mpBatches=1, lin_mols=True, **kargs):
    """
    Takes a function and parallelizes its calculations, lin_parts specifies how to split up df
    :param func: function to be parallelized
    :param pd_obj: tuple where [0] is argument used to pass molecules to callback fct, list of indivisible atoms
    which will be grouped into molecules, e.g. all dates, or all features (features for aux feat importance)
    :param numThreads:
    :param mpBatches:
    :param lin_mols:
    :param kargs:
    :return:
    """
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), numThreads * mpBatches) # splits into molecules
    else:
        parts = nested_parts(len(pd_obj[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]: parts[i]], 'func':func}  # job contains fct, and molecule of feature
        job.update(kargs) # combine the 2 dictionaries
        jobs.append(job)  # jobs list
        # all the jobs now set-up, callback function for each molecule and identical parametrization
    if numThreads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, numThreads=numThreads)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out

    for i in out:
        df0 = df0.append(i) # concatenating outputs

    return df0.sort_index()


def process_jobs_(jobs):
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)

    return out


def process_jobs(jobs, task=None, numThreads=8):
    if task is None:
        task = jobs[0]['func'].__name__ # name to print
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expand_call, jobs), [], time.time()
    for i, out_ in tqdm(enumerate(outputs, 1)):
        out.append(out_)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()
    return out


def report_progress(job_num, num_jobs, time0, task):
    msg = [float(job_num)/num_jobs, (time.time() - time0)/60]
    msg.append(msg[1] * (1/msg[0]-1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = time_stamp + ' ' + str(round(msg[0]*100, 2)) + '% ' + task + ' done after ' + str(
        round(msg[1], 2)) + '  minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    if job_num < num_jobs:
        sys.stderr.write(msg+ '\r')
    else:
        sys.stderr.write(msg + '\n')
    return


def process_jobs_redux(jobs, task=None, numThreads=8, redux=None, reduxArgs={}, reduxInPlace=False):
    """
    run in parallel, jobs must contain func callback, redux prevents wasting memory by reducing output on the fly
    :param jobs:
    :param task:
    :param numThreads:
    :param redux: callback to function that carries out the extension
    :param reduxArgs:
    :param reduxInPlace: whether redux should happen inplace or not (e.g. list.append is inplace as is dict update)
    :return:
    """
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    imap, out, time0 = pool.imap_unordered(expand_call, jobs), None, time.time()
    # processes asynchronous output
    for i, out_ in enumerate(imap, 1):
        if out is None:
            if redux is None:
                out, redux, reduxArgs = [out_], list.append, True
            else:
                import copy
                out = copy.deepcopy(out_)
        else:
            if reduxInPlace:
                redux(out, out_, **reduxArgs)
            else:
                out = redux(out, out_, **reduxArgs)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()
    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out

# enhanced mp_Pandas_obj...


def mp_job_list(func, argList, numThreads=8, mpBatches=1., linMols=True, redux=None, reduxArgs={}, reduxInPlace=False,
                **kargs):
    if linMols:
        parts = lin_parts(len(argList[1]), numThreads*mpBatches)
    else:
        parts = nested_parts(len(argList[1]), numThreads*mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {argList[0]: argList[1][parts[i-1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    out = process_jobs_redux(jobs, redux=redux, reduxArgs=reduxArgs, reduxInPlace=reduxInPlace, numThreads=numThreads)

    return out


def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms, num_threads, upper_triang=False):
    parts, num_threads_ = [0], min(num_threads, num_atoms)
    for num in range(num_threads_):
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + num_atoms *(num_atoms + 1)/num_threads_)
        part = (-1 + part**.5)/2
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triang:
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts