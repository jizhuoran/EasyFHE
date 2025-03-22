import torch


def auto_sync(func):
    def wrapper(*args, **kwargs):
        torch.cpu.synchronize()
        torch.cuda.synchronize()
        result = func(*args, **kwargs)
        torch.cpu.synchronize()
        torch.cuda.synchronize()
        return result
    return wrapper


def check_meta_equal(func):
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        in0, in1 = args[0], args[1]
        assert len(in0.cv) == len(in1.cv)
        assert in0.is_ext == in1.is_ext
        assert in0.slots == in1.slots
        
        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            assert in0.cur_limbs == in1.cur_limbs
            assert in0.scaling_factor == in1.scaling_factor
            assert in0.noise_deg == in1.noise_deg

        return func(*args, **kwargs)

    return wrapper
