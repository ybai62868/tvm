import tvm
import time
from tvm import te
from tvm import relay


from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
from gluoncv import data as gcv_data, model_zoo, utils
import mxnet as mx
import numpy as np

from tvm.relay.backend import compile_engine
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib import tensorrt


im_fname = download_testdata(
    "https://github.com/dmlc/web-data/blob/main/" + "gluoncv/detection/street_small.jpg?raw=true",
    "street_small.jpg",
    module="data",
)


def get_ssd_model(model_name, image_size=512):
    input_name = "data"
    input_shape = (1, 3, image_size, image_size)

    # Prepare model and input data
    data, img = gcv_data.transforms.presets.ssd.load_test(im_fname, short=image_size)

    # Prepare SSD model 
    block = model_zoo.get_model(model_name, pretrained=True)
    block.hybridize()
    block.forward(data)
    block.export("temp")


    model_json = mx.symbol.load("temp-symbol.json")
    save_dict = mx.ndarray.load("temp-0000.params")
    arg_params = {}
    aux_params = {}

    for param, val in save_dict.items():
        param_type, param_name = param.split(":", 1)
        if param_type == "arg":
            arg_params[param_name] = val
        elif param_type == "aux":
            aux_params[param_name] = val

    mod, params = relay.frontend.from_mxnet(model_json, 
                                           {input_name: input_shape}, arg_params=arg_params,
                                            aux_params=aux_params)
    
    return mod, params, block.classes, data.asnumpy(), img

mod, params, class_names, data, img = get_ssd_model("ssd_512_resnet50_v1_coco")

def profile_graph(func):
    class OpProfiler(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.ops = {}
        
        def visit_call(self, call):
            op = call.op
            if op not in self.ops:
                self.ops[op] = 0
            self.ops[op] += 1
            super().visit_call(call)

        def get_trt_graph_num(self):
            cnt = 0
            for op in self.ops:
                if str(op).find("tensorrt") != -1:
                    cnt += 1
            return cnt 

    profiler = OpProfiler()
    profiler.visit(func)
    print("Total number of operations: %d" % sum(profiler.ops.values()))
    print("Detail breadown")
    for op, count in profiler.ops.items():
        print("\t%s: %d" % (op, count))
    print("TensorRT subgraph #: %d " % profiler.get_trt_graph_num())

print("hello")
profile_graph(mod["main"])



def build_and_run(mod, data, params, build_config=None):
    compile_engine.get().clear()
    with tvm.transform.PassContext(opt_level=0, config=build_config):
        lib = relay.build(mod, target="cuda", params=params)

    # Create the runtime module
    mod = graph_runtime.GraphModule(lib["default"](tvm.gpu(0)))

    # Run inference 10 times
    times = []
    for _ in range(10):
        start = time.time()
        mod.run(data=data)
        times.append(time.time() - start)
    
    print("Runtime module structure: ")
    print("\t %s" % str(lib.get_lib()))
    for sub_mod in lib.get_lib().imported_modules:
        print("\t |- %s" % str(sub_mod))
    print("Median inference latency %.2f ms" % (1000 * np.median(times)))
    return mod, lib


_ = build_and_run(mod, data, params)

trt_mod, config = tensorrt.partition_for_tensorrt(mod, params)
print(config)

profile_graph(trt_mod["main"])
print("===============")
print(trt_mod["main"].astext(show_meta_data=False))

profile_graph(trt_mod["tensorrt_0"])
profile_graph(trt_mod["tensorrt_371"])

config["remove_no_mac_subgraphs"] = True
with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options":config}):
    trt_mod = tensorrt.prune_tensorrt_subgraphs(trt_mod)


profile_graph(trt_mod["main"])


print("helllo")
runtime_mod, lib = build_and_run(
    trt_mod, data, params, build_config={"relay.ext.tensorrt.options":config}
)


# print(lib.get_lib().imported_modules[1].get_source())

from matplotlib import pyplot as plt

results = [runtime_mod.get_output(i).asnumpy() for i in range(runtime_mod.get_num_outputs())]

ax = utils.viz.plot_bbox(
    img, results[2][0], results[1][0], results[0][0], class_names=class_names
)
plt.savefig("output.png")