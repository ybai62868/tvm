from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt_host = "llvm"
tgt = "cuda"

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

s = te.create_schedule(C.op)
ir_m = tvm.lower(s, [A, B, C], simple_mode=True,name='vadd')
rt_m = tvm.build(ir_m, [A, B, C], target='c', name='vadd')
print("tir:\n", ir_m.astext(show_meta_data=False))
print("source code:\n",rt_m.get_source())


# bx, tx = s[C].split(C.op.axis[0], factor=64)


# if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
#     s[C].bind(bx, te.thread_axis("blockIdx.x"))
#     s[C].bind(tx, te.thread_axis("threadIdx.x"))


# fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")


# ctx = tvm.context(tgt, 0)
# n = 1024
# a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
# b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
# c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
# fadd(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())


# if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
#     dev_module = fadd.imported_modules[0]
#     print("-----GPU code-----")
#     print(dev_module.get_source())
# else:
#     print(fadd.get_source())