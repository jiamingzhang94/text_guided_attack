from torch.profiler import profile, record_function, ProfilerActivity
import torch

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input)

# 保存为 .json 文件
prof.export_chrome_trace("path/to/your/trace.json")
