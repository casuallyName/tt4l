# @Time     : 2024/11/22 14:19
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :


VERSION = "0.0.4"


def get_info():
    import platform

    import accelerate
    import datasets
    import torch
    import transformers
    from transformers.utils import is_torch_cuda_available, is_torch_npu_available
    info = {
        "`tt4l` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
        "Accelerate version": accelerate.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann

    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass
    return info
