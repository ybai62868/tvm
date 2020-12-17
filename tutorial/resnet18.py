import tvm
from tvm import te

M = 1024
K = 1024
N = 1024

# Algorithm
k = te.reduce_axis((0, K), 'k')
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
           name='C')

# Default schedule
s = te.create_schedule(C.op)
ir_m = tvm.lower(s, [A, B, C], simple_mode=True,name='mmult')
rt_m = tvm.build(ir_m, [A, B, C], target='c', name='mmult')

# print tir
print("tir:\n", ir_m.astext(show_meta_data=False))
# print source code
print("source code:\n",rt_m.get_source())


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
import tvm

# Resnet18 workload
resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)

with relay.build_config(opt_level=0):
    _, resnet18_lib, _ = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)

# print relay ir
print(resnet18_mod.astext(show_meta_data=False))

# print source code
print(resnet18_lib.get_source())