#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import nvidia_smi

if __name__ == "__main__":
  nvidia_smi.nvmlInit()
  for gpu_id in range(8):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
