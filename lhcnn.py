import numpy as np
import math
import weakref
import contextlib

from PIL import Image


########################################
#                   FILE  DESCRIPTION
# File Name: Config.py
# Author: 李航 Lihang
# Description:
#    1、全局配置
#    2、主要用来控制自动梯度(backprop)的开关
#    
########################################
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

SETUP_VARIABLE = True






########################################
#                   FILE  DESCRIPTION
# File Name: variable.py
# Author: 李航 Lihang
# Description:
#    1、张量，类似于Tensor，实际上是numpy的多维数组
#    2、用于管理整个计算链条，自动计算反向传播和自动梯度
#    
########################################
class Variable:
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError( '{} is not supported'.format(type(data)) )
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        
    @property
    def shape(self):
        return self.data.shape
        
    @property
    def dtype(self):
        return self.data.dtype
        
    def __len__(self):
        return len(self.data)
        
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def cleargrad(self):
        self.grad = None

    def set_creator(self, func):
        self.creator = func
        #self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                #funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref
        
        
class Parameter(Variable):
    pass
    
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x







########################################
#                   FILE   DESCRIPTION
# File Name: Function.py
# Author: 李航 Lihang
# Description:
#    1、函数，Backend
#    2、不同于Layer中的同名类，Layer相当于前端
#    
########################################
class Function:
    def __call__(self,  *inputs):
        input_list = [ as_variable(x) for x in inputs]
        
        xs = [x.data for x in input_list]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        output_list = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            #self.generation = max([x.generation for x in inputs])
            for output in output_list:
                output.set_creator(self)
            self.inputs = input_list
            self.outputs = [weakref.ref(output) for output in output_list]
        
        return output_list if len(output_list) > 1 else output_list[0]
        
    def forward(self, xs):
        raise NotImplementedError()
        
    def backward(selfs, gys):
        raise NotImplementedError()
        
###############
# 四则运算
# Add Sub Mul Div else...
###############
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def call_add(x0, x1):
    return Add()(x0, x1)

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def call_sub(x0, x1):
    return Sub()(x0, x1)

def call_rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def call_neg(x):
    return Neg()(x)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def call_mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)
    


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def call_exp(x):
    return Exp()(x)

###############
# 矩阵操作
# Dot Transpose ELSE...
###############
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = call_matmul(gy, W.T)
        gW = call_matmul(x.T, gy)
        return gx, gW


def call_matmul(x, W):
    return MatMul()(x, W)
    

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return call_transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return call_transpose(gy, inv_axes)


def call_transpose(x, axes=None):
    return Transpose(axes)(x)


###############
# SumTo
# TODO...
###############
def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
    
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = call_broadcast_to(gy, self.x_shape)
        return gx


def call_sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy

class SumFn(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = call_broadcast_to(gy, self.x_shape)
        return gx


def call_sum(x, axis=None, keepdims=False):
    return SumFn(axis, keepdims)(x)

###############
# BroadcastTo
# TODO...
###############
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def call_broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

###############
# Reshape Fn
# TODO...
###############
class ReshapeFn(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return call_reshape(gy, self.x_shape)


def call_reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return ReshapeFn(shape)(x)

###############
# Linear Fn
# 1、神经元的基本单元
# 2、ELSE...
###############
class LinearFn(Function):
    def forward(self, x, W, b):
        y1 = x.dot(W)
        y = y1+b
        return y
        
    def backward(self,  gy):
        x, W, b = self.inputs
        gb = call_sum_to(gy, b.shape)
        gx = gy.dot(W.T())
        gW = x.T().dot(gy)
        return gx, gW, gb

###############
# MeanSquaredError Fn
# 1、均方差
# 2、ELSE...
###############
class MeanSquaredErrorFn(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
        
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = Sub()(x0, x1)
        gy = call_broadcast_to(gy, diff.shape) # 对标 .sum()
        gx0 = gy * 2.0 * diff * (1.0/len(diff)) # 除法来自于均值
        gx1 = -gx0
        return gx0, gx1

###############
# Tanh Fn
# 1、Tanh
# 2、ELSE...
###############
class TanhFn(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx

def call_tanh(x):
    return TanhFn()(x)

###############
# ReLU Fn
# 1、ReLU
# 2、ELSE...
###############
class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def call_relu(x):
    return ReLU()(x)
    
###############
# Softmax Fn
# 1、Softmax
# 2、ELSE...
###############
class SoftmaxFn(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def call_softmax(x):
    return SoftmaxFn()(x)
    

###############
# LogSoftmax Fn
# 1、LogSoftmax
# 2、ELSE...
###############
def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m

class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - call_exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx

    
###############
# LogSoftmax Fn
# 1、LogSoftmax
# 2、ELSE...
###############
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = call_softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y



###############
# SETUP_VARIABLE
# 1、Variable 重载运算符
# 2、ELSE...
###############
if SETUP_VARIABLE:
    Variable.__add__ = call_add
    Variable.__sub__ = call_sub
    Variable.__rsub__ = call_rsub
    Variable.__mul__ = call_mul
    Variable.__neg__ = call_neg
    Variable.dot = call_matmul
    Variable.sum = call_sum
    Variable.T = call_transpose






########################################
#                   FILE   DESCRIPTION
# File Name: Layer
# Author: 李航 Lihang
# Description:
#    1、网络层
#    2、代表一层神经元的组织结构
#    3、具体运算是调用Backend 的Function中的算法，以便自动计算反向传播
#
########################################
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
        
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]        
        
    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

###############
# Linear
# 1、全连接神经元层
# 2、ELSE...
###############
class Linear(Layer):
    def __init__(self,  in_size,  out_size ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = Parameter(None, name='W')
        self._init_W()

        self.b = Parameter(np.zeros(out_size, dtype=np.float32), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(np.float32) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        Fn = LinearFn()
        y = Fn(x, self.W, self.b)
        return y

###############
# Sequential
# 1、队列容器
# 2、ELSE...
###############
class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
###############
# Optimizer
# 1、优化器基类
# 2、ELSE...
###############
class Optimizer:
    def __init__(self):
        self.target = None

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

###############
# Optimizer
# 1、SGD优化器
# 2、ELSE...
###############
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
        







########################################
#                   FILE   DESCRIPTION
# File Name: Conv2d
# Author: 李航 Lihang
# Description:
#    1、Conv2d
#    2、ELSE..
#
########################################
def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p

def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W()

        y = call_conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Conv2dFn(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):

        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = np.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # ==== gx ====
        gx = call_deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb
        
def call_conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2dFn(stride, pad)(x, W, b)

class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = np.tensordot(Weight, x, (0, 1))
        gcol = np.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = call_conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb

def call_deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = call_deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = call_conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    img = np.pad(img,
                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                   dtype=col.dtype)
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
    return img[:, :, PH:H + PH, PW:W + PW]


class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + np.arange(0, self.indexes.size * KH * KW, KH * KW))
        
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = np.swapaxes(gcol, 2, 4)
        gcol = np.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)





########################################
#                   FILE   DESCRIPTION
# File Name: DataLoader
# Author: 李航 Lihang
# Description:
#    1、数据集处理
#    2、DataLoader
#    3、ELSE..
#
########################################

class Compose:
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


class ToArray:
    """Convert PIL Image to NumPy array."""
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class Flatten:
    """Flatten a NumPy array.
    """
    def __call__(self, array):
        return array.flatten()


class Normalize:
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


###############
# DataLoader
# 1、打乱、批量
# 2、ELSE...
###############
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()


