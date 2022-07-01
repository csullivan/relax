# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import topi
import tvm.topi.testing
import tvm.relay.op.strategy
from tvm import autotvm, relay, relax, te
from tvm.relax.testing import relay_translator
from tvm.meta_schedule import ApplyHistoryBest
from tvm.meta_schedule.testing.utils import apply_fixed_schedules
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.script import tir as T

from tvm.relax.testing import relay_translator


def compute_tir_conv2d_nchw_oihw(data_shape, weight_shape, dtype):
    assert dtype == "float32"
    OC, IC, FH, FW = weight_shape

    padding = (0, 0, 0, 0)
    strides = (1, 1)
    dilation = (1, 1)
    output_shape = (
        data_shape[0],
        weight_shape[0],
        (data_shape[2] - ((weight_shape[2] - 1) * dilation[0] + 1) + padding[0] + padding[1])
        // strides[0]
        + 1,
        (data_shape[3] - ((weight_shape[3] - 1) * dilation[1] + 1) + padding[2] + padding[3])
        // strides[1]
        + 1,
    )
    N, K, BH, BW = output_shape

    # fmt: off
    @T.prim_func
    def conv2d(a: T.handle, filt: T.handle, b: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, data_shape, dtype=dtype)
        Filter = T.match_buffer(filt, weight_shape, dtype=dtype)
        B = T.match_buffer(b, output_shape, dtype=dtype)
        for n, k, bh, bw in T.grid(N, K, BH, BW):
            with T.block("init"):
                vn, vk, vbh, vbw = T.axis.remap("SSSS", [n, k, bh, bw])
                B[vn, vk, vbh, vbw] = T.float32(0)
            for ic, fh, fw in T.grid(IC, FH, FW):
                with T.block("update"):
                    vn, vk, vbh, vbw, vc, vfh, vfw = T.axis.remap("SSSSRRR", [n, k, bh, bw, ic, fh, fw])
                    B[vn, vk, vbh, vbw] = B[vn, vk, vbh, vbw] + A[vn, vc, vbh + vfh, vbw + vfw] * Filter[vk, vc, vfh, vfw]
    # fmt: on

    return conv2d


def schedule_tir_conv2d_nchw_oihw(sch):
    update_block = sch.get_block("update")
    vn, vk, vbh, vbw, vc, vfh, vfw = sch.get_loops(update_block)
    sch.split(vk, factors=(None, 32))


@autotvm.register_topi_compute("test/conv2d_1")
def _compute_conv2d_1(cfg, input, filter, strides, padding, dilation, out_dtype):
    prim_func = compute_tir_conv2d_nchw_oihw(input.shape, filter.shape, input.dtype)
    output = te.extern_primfunc([input, filter], prim_func, name="tir")
    return output


@autotvm.register_topi_schedule("test/conv2d_1")
def _schedule_conv2d_1(cfg, outs):
    s = te.create_schedule([x.op for x in outs])
    return s


@tvm.target.override_native_generic_func("test_conv2d_strategy_tir")
def _tmp_strategy_tir_impl(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    if attrs.groups == 1 and attrs.data_layout == "NCHW" and attrs.kernel_layout == "OIHW":
        strategy.add_implementation(
            relay.op.strategy.wrap_compute_conv2d(_compute_conv2d_1),
            relay.op.strategy.wrap_topi_schedule(_schedule_conv2d_1),
            name="conv2d_2",
            plevel=15,
        )
    else:
        raise ValueError("No valid strategy found")
    return strategy


@tvm.target.override_native_generic_func("test_conv2d_strategy_topi_generic")
def _tmp_strategy_te_topi_generic_impl(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    if attrs.groups == 1 and attrs.data_layout == "NCHW" and attrs.kernel_layout == "OIHW":
        strategy.add_implementation(
            tvm.relay.op.strategy.wrap_compute_conv2d(topi.nn.conv2d_nchw),
            tvm.relay.op.strategy.wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),
            name="conv2d_2",
            plevel=15,
        )
    else:
        raise ValueError("No valid strategy found")
    return strategy


def get_conv2d(data_shape, weight_shape, dtype, **kwargs):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)
    conv2d = relay.nn.conv2d(
        data,
        weight,
        **kwargs,
    )
    return relay.Function([data, weight], conv2d)


def get_ref(data, weight, stride, padding):
    return tvm.topi.testing.conv2d_nchw_python(data, weight, stride, padding)


class Conv2dBase:
    FW = tvm.testing.parameter(3)
    FH = tvm.testing.parameter(3)
    IC = tvm.testing.parameter(64)
    OC = tvm.testing.parameter(128)
    W = tvm.testing.parameter(56)
    H = tvm.testing.parameter(56)
    N = tvm.testing.parameter(1)
    padding = tvm.testing.parameter((0, 0))
    strides = tvm.testing.parameter((1, 1))
    dtype = tvm.testing.parameter("float32")


@pytest.fixture
def data_shape(N, IC, H, W):
    return (N, IC, H, W)


@pytest.fixture
def weight_shape(OC, IC, FH, FW):
    return (OC, IC, FH, FW)


@pytest.fixture
def conv2d_relay_mod(data_shape, weight_shape, padding, strides, dtype):
    return tvm.IRModule.from_expr(
        get_conv2d(
            data_shape,
            weight_shape,
            dtype,
            padding=padding,
            strides=strides,
            channels=weight_shape[0],
            kernel_size=weight_shape[2:],
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
    )


@pytest.fixture
def conv2d_numpy_tensors(data_shape, weight_shape, dtype):
    data_np = np.random.randn(*data_shape).astype(dtype)
    weight_np = np.random.randn(*weight_shape).astype(dtype)
    return {"data": data_np, "weight": weight_np}


def conv2d_tir_translator(data_shape, weight_shape, dtype):
    conv2d_primfunc = compute_tir_conv2d_nchw_oihw(data_shape, weight_shape, dtype)
    return {"nn.conv2d": conv2d_primfunc}


class TestConv2d(Conv2dBase):
    tir_translator = tvm.testing.parameter(None, conv2d_tir_translator)

    def test_conv2d(
        self,
        data_shape,
        weight_shape,
        padding,
        strides,
        dtype,
        conv2d_relay_mod,
        conv2d_numpy_tensors,
        tir_translator,
    ):
        target = "llvm"
        with TempOpAttr("nn.conv2d", "FTVMStrategy", _tmp_strategy_te_topi_generic_impl):
            op_tir_translation = None
            if tir_translator:
                op_tir_translation = tir_translator(data_shape, weight_shape, dtype)
            relax_mod = relay_translator.from_relay(
                conv2d_relay_mod["main"], target, translate_op_with_tir=op_tir_translation
            )
            print(relax_mod.script())
            ex = relax.vm.build(relax_mod, target)

        dev = tvm.cpu()
        runtime = relax.VirtualMachine(ex, dev)

        runtime.set_input("main", **conv2d_numpy_tensors)
        out = runtime["main"]()
        ref = get_ref(
            conv2d_numpy_tensors["data"], conv2d_numpy_tensors["weight"], strides, padding
        )
        tvm.testing.assert_allclose(out.numpy(), ref, atol=1e-4, rtol=1e-4)

    def test_conv2d_relay_te(
        self,
        data_shape,
        weight_shape,
        padding,
        strides,
        dtype,
        conv2d_relay_mod,
        conv2d_numpy_tensors,
    ):
        target = "llvm"
        with TempOpAttr("nn.conv2d", "FTVMStrategy", _tmp_strategy_te_topi_generic_impl):
            with tvm.transform.PassContext(
                opt_level=3,
            ):
                lib = relay.build(conv2d_relay_mod, target=target)

        dev = tvm.device(target, 0)

        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        runtime.set_input(**conv2d_numpy_tensors)
        runtime.run()

        out = runtime.get_output(0).numpy()

        ref = get_ref(
            conv2d_numpy_tensors["data"], conv2d_numpy_tensors["weight"], strides, padding
        )

        tvm.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_conv2d()
