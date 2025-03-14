import ctypes
from ctypes import c_uint8, c_uint16, c_uint32, c_uint64, c_int32, c_int64, c_void_p, c_char_p, POINTER, Structure, byref, create_string_buffer
from types import SimpleNamespace
import numpy as np

class HalideType(Structure):
    _fields_ = [
        ("code", c_uint8),
        ("bits", c_uint8),
        ("lanes", c_uint16)
    ]

class HalideDimension(Structure):
    _fields_ = [
        ("min", c_int32),
        ("extent", c_int32),
        ("stride", c_int32),
        ("flags", c_uint32)
    ]

class HalideBuffer(Structure):
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
    def __init__(self, width, height, channels):
        self.buffer = HalideBuffer()
        self.array = (c_uint8 * (width * height * channels))()

        self.buffer.dev = 0
        self.buffer.device_interface = None
        self.buffer.host = self.array
        self.buffer.flags = 0
        self.buffer.type = HalideType(1, 8, 1) # code, bits, lanes
        self.buffer.dimensions = 3

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
        return self.buffer.dimension[2].stride

    @property
    def channels(self):
        return self.buffer.dimension[2].extent

    def fill_with_checkerboard(self, size):
        width = self.width
        array = self.array
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
        np_whc = np.ctypeslib.as_array(self.array).reshape(self.width, self.height, self.channels)
        return np.transpose(np_whc, (1, 0, 2))
    
    def from_numpy(self, np_array_hwc: np.ndarray):
        assert np_array_hwc.dtype == np.uint8
        assert np_array_hwc.shape == (self.height, self.width, self.channels)
        np_array_whc = np.transpose(np_array_hwc, (1, 0, 2)).copy()
        ctypes.memmove(ctypes.byref(self.array), np_array_whc.ctypes.data, np_array_whc.nbytes)

def make_buffer(width, height, channels):
    return Buffer(width, height, channels)

def make_dereferencer(type, bits):
    if bits == 1:
        return lambda buf: None if not buf else bool(ctypes.cast(buf, POINTER(ctypes.c_int8))[0])
    elif type == "int" and bits == 8:
        reader = ctypes.c_int8
    elif type == "uint" and bits == 8:
        reader = ctypes.c_uint8
    elif type == "int" and bits % 8 == 0:
        reader = getattr(ctypes, f"c_int{bits}")
    elif type == "uint" and bits % 8 == 0:
        reader = getattr(ctypes, f"c_uint{bits}")
    elif type == "float" and bits == 32:
        reader = ctypes.c_float
    elif type == "float" and bits == 64:
        reader = ctypes.c_double
    else:
        raise ValueError(f"invalid type: {type} {bits}")

    return lambda buf: None if not buf else ctypes.cast(buf, POINTER(reader))[0]

def make_caster(type, bits):
    if bits == 1:
        return lambda buf: None if not buf else bool(ctypes.cast(buf, POINTER(ctypes.c_int8))[0])
    elif type == "int" and bits == 8:
        reader = ctypes.c_int8
    elif type == "uint" and bits == 8:
        reader = ctypes.c_uint8
    elif type == "int" and bits % 8 == 0:
        reader = getattr(ctypes, f"c_int{bits}")
    elif type == "uint" and bits % 8 == 0:
        reader = getattr(ctypes, f"c_uint{bits}")
    elif type == "float" and bits == 32:
        reader = ctypes.c_float
    elif type == "float" and bits == 64:
        reader = ctypes.c_double
    else:
        raise ValueError(f"invalid type: {type} {bits}")

    return lambda buf: None if not buf else reader(buf)

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

    for i in range(nargs):
        ma = convert_argument_struct(args[i])

        supported_buffer = ma["dimensions"] == 3 and ma["is_int"] and ma["bits"] == 8

        if ma["type"] == "handle":
            if i != 0:
                raise ValueError(f"Unexpected handle at position other than 0: {i}")

            params.append(c_void_p)
        elif ma["kind"] == "output" and supported_buffer:
            params.append(POINTER(HalideBuffer))
            outputs += 1
        elif ma["kind"] == "input" and supported_buffer:
            params.append(POINTER(HalideBuffer))
        elif ma["kind"] == "scalar":
            if ma["type"] == "float" and ma["bits"] == 32:
                params.append(ctypes.c_float)
            elif ma["type"] == "float" and ma["bits"] == 64:
                params.append(ctypes.c_double)
            elif ma["is_int"] and ma["bits"] > 0 and ma["bits"] % 8 == 0:
                params.append(getattr(ctypes, f"c_{ma['type']}{ma['bits']}"))
            else:
                raise ValueError(f"Unhandled type: {ma['type']} {ma['bits']}")
        else:
            raise ValueError(f"Unhandled kind: {ma['kind']} type: {ma['type']} with {ma['bits']} bits and {ma['dimensions']} dimensions")

    if outputs != 1:
        raise ValueError(f"Expected exactly one output, got: {outputs}")

    return params

def gather_vars(args, nargs):
    vars = []

    for i in range(nargs):
        ma = convert_argument_struct(args[i])

        if ma["kind"] == "input":
            vars.append(SimpleNamespace(**{
                "name": ma["name"],
                "make_buffer": make_buffer,
                "buffer": True,
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
        if self.render_library:
            handle = self.render_library._handle
            del self.render_library
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            self.render_library = None
            self.render_function = None

    def call(self):
        if self.render_function:
            return self.render_function()
        else:
            return None, "No currently bound function."

    def prepare(self, width, height, channels):
        curr = self.output_buffer
        if curr and curr.width == width and curr.height == height and curr.channels == channels:
            print('LibBinder.prepare(): reusing buffer')
            return
        self.output_buffer = make_buffer(width, height, channels) # [halide buffer ptr, native buffer]

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

        params = gather_params(metadata.arguments, metadata.num_arguments)
        vars = gather_vars(metadata.arguments, metadata.num_arguments)

        boundfn = ctypes.CFUNCTYPE(ctypes.c_int, *params)(rawfn)

        error_buffer = create_string_buffer(4096)
        was_error = True

        def render_function():
            nonlocal was_error
            if was_error:
                error_buffer.value = b'\0' * 4096

            result = [None] #[byref(error_buffer)]
            for arg in vars:
                result.append(args[arg.name])
            result.append(self.output_buffer.buffer)

            code = boundfn(*result)
            was_error = code != 0

            if code == 0:
                return self.output_buffer, error_buffer
            else:
                return None, error_buffer

        self.render_function = render_function

        return vars