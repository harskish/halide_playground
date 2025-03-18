import ctypes
from ctypes import c_float, c_uint8, c_uint16, c_uint32, c_uint64, c_int32, c_int64, c_void_p, c_char_p, POINTER, Structure, byref, create_string_buffer
from types import SimpleNamespace
import platform
import numpy as np

hl_type_codes = {
    'int': 0,     # signed integers
    'uint': 1,    # unsigned integers
    'float': 2,   # IEEE floating point numbers
    'handle': 3,  # opaque pointer type (void *)
    'bfloat': 4,  # bfloat 16-bit format
}

class HalideType(Structure):
    _fields_ = [
        ("code", c_uint8),  # See hl_type_codes above
        ("bits", c_uint8),  # Bits of precision of a single scalar value of this type
        ("lanes", c_uint16) # How many elements in a vector, 1 for scalar types
    ]

class HalideDimension(Structure):
    _fields_ = [
        ("min", c_int32),
        ("extent", c_int32),
        ("stride", c_int32),
        ("flags", c_uint32)
    ]

# TODO: single buffer type with void* host?
class HalideBufferU8(Structure):
    _fields_ = [
        ("dev", c_uint64),
        ("device_interface", c_void_p),
        ("host", POINTER(c_uint8)),
        ("flags", c_uint64),
        ("type", HalideType),
        ("dimensions", c_int32),
        ("dimension", POINTER(HalideDimension)),
        ("padding", c_void_p)
    ]

class HalideBufferU16(Structure):
    _fields_ = [
        ("dev", c_uint64),
        ("device_interface", c_void_p),
        ("host", POINTER(c_uint16)),
        ("flags", c_uint64),
        ("type", HalideType),
        ("dimensions", c_int32),
        ("dimension", POINTER(HalideDimension)),
        ("padding", c_void_p)
    ]

# https://github.com/halide/Halide/commit/d2d2f846ed51721d2cc1679b4b3e95b315d03e67
class HalideFilterArgument(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("kind", c_int32),
        ("dimensions", c_int32),
        ("type", HalideType),
        ("def", c_void_p),
        ("min", c_void_p),
        ("max", c_void_p),
        ("estimate", c_void_p),
        ("buffer_estimates", POINTER(POINTER(c_int64))) # array of dimensions*2 pointers of (min, extent)
    ]

class HalideFilterMetadata(Structure):
    _fields_ = [
        ("version", c_int32),
        ("num_arguments", c_int32),
        ("arguments", POINTER(HalideFilterArgument)),
        ("target", c_char_p),
        ("name", c_char_p)
    ]

class Buffer:
    def __init__(self, width, height, channels, dtype):
        assert issubclass(dtype, ctypes._SimpleCData), 'dtype must be a ctypes type'
        buffer_type = { c_uint8: HalideBufferU8, c_uint16: HalideBufferU16 }[dtype]
        self.buffer = buffer_type()
        self.array = (dtype * (width * height * channels))()
        self.dtype = dtype

        self.buffer.dev = 0
        self.buffer.device_interface = None
        self.buffer.host = self.array
        self.buffer.flags = 0
        self.buffer.type = HalideType(hl_type_codes['uint'], ctypes.sizeof(dtype) * 8, 1) # code, bits, lanes
        self.buffer.dimensions = 3

        # https://github.com/halide/atom/blob/523238d9/lib/halide-lib-binder.coffee#L67
        self.buffer.dimension = (HalideDimension * 3)()
        self.buffer.dimension[0] = HalideDimension(0, width, 1, 0) # min, extent, stride, flags
        self.buffer.dimension[1] = HalideDimension(0, height, width, 0)
        self.buffer.dimension[2] = HalideDimension(0, channels, width * height, 0)

    @property
    def width(self):
        return self.buffer.dimension[1].stride
    
    @property
    def height(self):
        return self.buffer.dimension[1].extent
    
    @property
    def plane(self):
        """Size of width*height single-color plane"""
        return self.buffer.dimension[2].stride

    @property
    def channels(self):
        return self.buffer.dimension[2].extent

    # https://github.com/halide/atom/blob/523238d9/lib/halide-lib-binder.coffee#L101
    def fill_with_checkerboard(self, size):
        width = self.width
        array = self.array # 1d array, x-axis has stride 1, y-axis has stride width
        plane = self.plane

        limit = min(width * size * 2, plane)
        for i in range(limit):
            y = i % width
            x = (i - y) // width
            if (x // size) % 2 != (y // size) % 2:
                array[i] = 64
            else:
                array[i] = 192

        for i in range(limit, plane):
            array[i] = array[i - limit]

        for i in range(plane, len(array)):
            array[i] = array[i - plane]
    
    def numpy_hwc(self) -> np.ndarray:
        # Pipeline output: x-axis stride 1, y-axis stride width
        np_chw = np.ctypeslib.as_array(self.array).reshape(self.channels, self.height, self.width)
        return np.transpose(np_chw, (1, 2, 0))
    
    def from_numpy(self, np_array_hwc: np.ndarray):
        # "By default halide assumes the first dimension (x in this case) is dense in memory (stride 1)"
        # https://github.com/harskish/anyscale/blob/b29a5c01/lib/halide_ops/halide_pt_op.py#L51
        #assert np_array_chw.ndim != 3 or np_array_chw.shape[0] in [1, 3, 4], 'Halide expects CHW tensors'
        assert np.ctypeslib.as_ctypes_type(np_array_hwc.dtype) == self.dtype
        assert np_array_hwc.shape == (self.height, self.width, self.channels)
        np_array_chw = np.transpose(np_array_hwc, (2, 0, 1)).copy()
        ctypes.memmove(ctypes.byref(self.array), np_array_chw.ctypes.data, np_array_chw.nbytes)

def make_buffer(width, height, channels, dtype):
    return Buffer(width, height, channels, dtype)

def make_dtype(type, bits):
    if type == "int" and bits == 8:
        return ctypes.c_int8
    elif type == "uint" and bits == 8:
        return ctypes.c_uint8
    elif type == "int" and bits % 8 == 0:
        return getattr(ctypes, f"c_int{bits}")
    elif type == "uint" and bits % 8 == 0:
        return getattr(ctypes, f"c_uint{bits}")
    elif type == "float" and bits == 32:
        return ctypes.c_float
    elif type == "float" and bits == 64:
        return ctypes.c_double
    else:
        raise ValueError(f"invalid type: {type} {bits}")

def make_dereferencer(type, bits):
    if bits == 1:
        return lambda buf: None if not buf else bool(ctypes.cast(buf, POINTER(ctypes.c_int8))[0])
    
    reader = make_dtype(type, bits)
    return lambda buf: None if not buf else ctypes.cast(buf, POINTER(reader))[0]

def make_caster(type, bits):
    if bits == 1:
        return lambda buf: None if not buf else bool(ctypes.cast(buf, POINTER(ctypes.c_int8))[0])
    
    dtype = make_dtype(type, bits)
    return lambda buf: dtype(buf)

def convert_argument_struct(ma):
    type = ["int", "uint", "float", "handle"][ma.type.code]

    return {
        "name": ma.name.decode(),
        "dimensions": ma.dimensions,
        "kind": ["scalar", "input", "output"][ma.kind],
        "type": type,
        "bits": ma.type.bits,
        "is_int": type in ["uint", "int"]
    }

def gather_params(args, nargs):
    params = []
    outputs = 0
    out_dtype = None

    for i in range(nargs):
        ma = convert_argument_struct(args[i])
        
        buffer_types = { 8: HalideBufferU8, 16: HalideBufferU16 }
        supported_buffer = ma["dimensions"] == 3 and ma["is_int"] and ma["bits"] in [8, 16]

        if ma["type"] == "handle":
            if i != 0:
                raise ValueError(f"Unexpected handle at position other than 0: {i}")

            params.append(c_void_p)
        elif ma["kind"] == "output" and supported_buffer:
            params.append(POINTER(buffer_types[ma["bits"]]))
            out_dtype = make_dtype(ma["type"], ma["bits"])
            outputs += 1
        elif ma["kind"] == "input" and supported_buffer:
            params.append(POINTER(buffer_types[ma["bits"]]))
        elif ma["kind"] == "scalar":
            if ma["type"] == "float" and ma["bits"] == 32:
                assert ctypes.sizeof(ctypes.c_float) == 32 // 8
                params.append(ctypes.c_float)
            elif ma["type"] == "float" and ma["bits"] == 64:
                assert ctypes.sizeof(ctypes.c_double) == 64 // 8
                params.append(ctypes.c_double)
            elif ma["is_int"] and ma["bits"] > 0 and ma["bits"] % 8 == 0:
                ctype = getattr(ctypes, f"c_{ma['type']}{ma['bits']}")
                assert ctypes.sizeof(ctype) == ma["bits"] // 8
                params.append(ctype)
            else:
                raise ValueError(f"Unhandled type: {ma['type']} {ma['bits']}")
        else:
            raise ValueError(f"Unhandled kind: {ma['kind']} type: {ma['type']} with {ma['bits']} bits and {ma['dimensions']} dimensions")

    if outputs != 1:
        raise ValueError(f"Expected exactly one output, got: {outputs}")

    return params, out_dtype

def gather_vars(args, nargs):
    vars = []

    for i in range(nargs):
        ma = convert_argument_struct(args[i])

        if ma["kind"] == "input":
            vars.append(SimpleNamespace(**{
                "name": ma["name"],
                "make_buffer": make_buffer,
                "buffer": True,
                "dtype": make_dtype(ma["type"], ma["bits"]),
            }))
        elif ma["kind"] == "scalar" and ma["type"] != "handle":
            dereffer = make_dereferencer(ma["type"], ma["bits"])
            caster = make_caster(ma["type"], ma["bits"])
            meta = SimpleNamespace(**{
                "int": ma["is_int"],
                "bool": ma["bits"] == 1,
                "name": ma["name"],
                "default": dereffer(getattr(args[i], 'def')),
                "min": dereffer(getattr(args[i], 'min')),
                "max": dereffer(getattr(args[i], 'max')),
                "type": ma["type"],
                "bits": ma["bits"],
                "buffer": False,
                "cast_fun": caster,
            })
            vars.append(meta)

    return vars

class LibBinder:
    def __init__(self):
        self.render_library = None
        self.render_function = None
        #self.output = None
        #self.outbuf = None
        self.output_buffer = None

    def close(self):
        if not self.render_library:
            return
        
        handle = self.render_library._handle
        del self.render_library
        self.render_library = None
        self.render_function = None
        
        if platform.system() == "Windows":    
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
        elif platform.system() == 'Darwin':
            assert ctypes.cdll.LoadLibrary("libSystem.dylib").dlclose(handle) == 0
        elif platform.system() == 'Linux':
            ctypes.cdll.LoadLibrary("libdl.so").dlclose(handle)
        else:
            raise ValueError(f"Unsupported platform: {platform.system()}")

    def call(self):
        if self.render_function:
            return self.render_function()
        else:
            return None, "No currently bound function."

    def prepare(self, width, height, channels, dtype):
        curr = self.output_buffer
        if curr and curr.width == width and curr.height == height and curr.channels == channels and dtype == curr.dtype:
            return
        print('LibBinder.prepare(): allocating new buffer')
        self.output_buffer = make_buffer(width, height, channels, dtype) # [halide buffer ptr, native buffer]

    def bind(self, fnname, libpath, args: dict):        
        self.close()

        self.render_library = ctypes.CDLL(libpath) # RTLD_NOW: always added on Unix-like
        rawfn = getattr(self.render_library, fnname)

        rawmetafn = getattr(self.render_library, fnname + "_metadata")
        rawmetafn.argtypes = []
        rawmetafn.restype = POINTER(HalideFilterMetadata)
        metadata = rawmetafn().contents

        # https://github.com/halide/Halide/commit/d2d2f846ed51721d2cc1679b4b3e95b315d03e67
        if metadata.version != 1:
            raise ValueError(f"Unknown Filter Metadata version: {metadata.version}")

        params, output_dtype = gather_params(metadata.arguments, metadata.num_arguments)
        vars = gather_vars(metadata.arguments, metadata.num_arguments)

        #assert ctypes.sizeof(ctypes.c_void_p) == 8
        #params[0] = POINTER(ctypes.c_char)

        rawfn.restype = ctypes.c_int
        rawfn.argtypes = params

        error_buffer = (ctypes.c_char * 4096)()
        #error_buffer = create_string_buffer(4096) # ctypes array of c_char.
        was_error = True

        def render_function():
            nonlocal was_error
            if was_error:
                error_buffer.value = b'\0' * 4096

            result = [error_buffer] # byref: (((char *)&obj) + offset)
            for arg in vars:
                result.append(args[arg.name])
            result.append(self.output_buffer.buffer)

            assert len(result) == len(params)
            code = rawfn(*result)
            was_error = code != 0

            err_str = error_buffer.value.decode()
            if code == 0:
                return self.output_buffer, err_str
            else:
                return None, err_str

        self.render_function = render_function

        return vars, output_dtype