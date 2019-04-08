# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:03:05 2019

@author: User
"""
import numpy as np
import threading
import time
import librosa


def powerNormalization(first_signal, window=400, overlap=0.75):
    '''
    Function for signal power normalization.

    This function normalize signal on maximum average power in windows.
    Input:
        first_signal - array like, first signal
        window - int, window size for averaging
        overlap - float, between 0 and 1, overlap factor for window
    Output:
        first_signal_norm - np.array, normalized first signal
    '''
    first_signal = np.array(first_signal)
    hop = int((1 - overlap) * window)
    # calculating maximum average power in all windows
    first_pow = max([np.mean(first_signal[i: i + window] ** 2) for i
                    in range(0, len(first_signal) - window + hop, hop)])
    if first_pow != 0:
        first_signal_norm = np.array(first_signal) / np.sqrt(first_pow)
        return first_signal_norm
    elif first_pow == 0:
        first_signal_norm = np.array(first_signal)
        return first_signal_norm


def fileWriter(sig, i):
    '''
    Function write signal into .wav file and normalize it.

    Input:
        sig - array like, signal
        i - int, number of signal
    '''
    for x in sig:
        y = powerNormalization(x)
        buf = np.reshape(np.array(y), (len(y), 1))
        librosa.output.write_wav(str(i) + '.wav', buf, sr=8000)
        i += 1


def parWriter(sigArr, nums=16):
    '''
    Function write signals into .wav files in nums threads.

    Input:
        sigArr - array, array of signals
        nums - int, number of threads
    '''
    size = len(sigArr) // nums
    # create array for threads
    thr = []
    for j in range(nums):
        # creating thread
        thr.append(threading.Thread(target=fileWriter,
                                    args=(sigArr[j * size: (j + 1) * size],
                                                 j * size,)))
        # starting thread
        thr[j].start()
    for j in range(nums):
        thr[j].join()


def main():
    # setting the length of signal
    seqLen = 10000
    nums = 2
    # setting seed
    np.random.seed(2)
    # creating signal
    sigArr = []
    for i in range(100):
        sigArr.append(np.random.randn(seqLen))
    start = time.time()
    for i in range(10):
        # in nums threads
        parWriter(sigArr, nums)
        # in one thread
        # fileWriter(sigArr, 0)
    end = time.time()
    print(end - start)
    procArr = [1, 2, 4, 8]
    timeArr = [2.86, 1.80, 1.65, 1.56]
    return 0


if __name__ == '__main__':
    main()
