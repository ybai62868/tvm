import tvm
from tvm import relay
import mxnet
from mxnet.gluon.model_zoo.vision import get_model


def get_demo_mod():
    # Loop
    iter1 = relay.var("iter1", shape = (), dtype = "int32")
    cond = relay.less(iter1, relay.const(2, dtype = "int32"))
    inc = iter1 + relay.const(1, dtype = "int32")
    loop_var = relay.var("while_loop")

    # Loop body
    d1 = relay.var("d1", shape = (1, 32, 56, 56), dtype="float32")
    w1 = relay.var("w1", shape = (32, 32, 3, 3), dtype="float32")
    b1 = relay.var("b1", shape=(32, ), dtype = "float32")
    conv = relay.nn.conv2d(d1, w1, strides=(1, 1), padding=(1, 1))
    bias = relay.nn.bias_add(conv, b1)
    relu = relay.nn.relu(bias)
    loop_cond_out = loop_var(inc, relu, w1, b1)

    conv = relay.nn.conv2d(d1, w1, strides=(1, 1), padding=(1, 1))
    bias = relay.nn.bias_add(conv, b1)
    relu = relay.nn.relu(bias)
    loop_break_out = relay.reshape(relu, (1, 56, 56, 32))

    ife = relay.If(cond, loop_cond_out, loop_break_out)

    data = relay.var("data", shape=(1, 32, 56, 56), dtype="float32")
    weight = relay.var("weight", shape=(32, 32, 3, 3), dtype="float32")
    bias = relay.var("bias", shape=(32,), dtype="float32")
    loop_func = relay.Function([iter1, d1, w1, b1], ife)

    out = relay.Let(loop_var, loop_func, loop_var(relay.const(0, dtype="int32"), data, weight, bias))

    func = relay.Function([data, weight, bias], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod

mod = get_demo_mod()
print(mod["main"].astext(show_meta_data=False))



demo_target = "byoc-target"

@tvm.ir.register_op_attr("reshape", "target.byoc-target")
def reshape(expr):
    return True

@tvm.ir.register_op_attr("add", "target.byoc-target")
def add(expr):
    return True


# Pattern-based annotation rules
def make_pattern(with_bias=True):
    from tvm.relay.dataflow_pattern import is_op, wildcard
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("nn.bias_add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)

conv2d_bias_relu_pat = ("byoc-target.conv2d_relu_with_bias", make_pattern(with_bias=True))
conv2d_relu_pat = ("byoc-target.conv2d_relu_wo_bias", make_pattern(with_bias=False))
patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]


mod2 = relay.transform.MergeComposite(patterns)(mod)
print(mod2["main"].astext(show_meta_data=False))



mod3 = relay.transform.AnnotateTarget("byoc-target")(mod2)
print(mod3["main"].astext(show_meta_data=False))

mod4 = relay.transform.MergeCompilerRegions()(mod3)
print(mod4["main"].astext(show_meta_data=False))

mod5 = relay.transform.PartitionGraph()(mod4)
print(mod5["main"].astext(show_meta_data=False))

for name in ["byoc-target_0", "byoc-target_2", "byoc-target_5"]:
    print("%s: " % name)
    print(mod5[name].astext(show_meta_data=False))
    print("===============================")