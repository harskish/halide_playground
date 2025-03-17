import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
import tempfile
import halide as hl
from pathlib import Path
import subprocess
from IPython.display import display
from IPython.core.display import HTML

assert (
    "EMSDK" in os.environ
), "Could not find `EMSDK` in the environment variables. You need to install Emscripten and point `EMSDK` to the installation directory before importing this module."


@dataclass
class Slider:
    param: hl.Param
    min_value: float
    max_value: float
    step: float
    initial_value: float = field(init=False)

    def __post_init__(self):
        tmp = hl.Func("tmp")
        tmp[()] = self.param
        self.initial_value = tmp.realize()[()]


def _halide_param_to_cpp_argument(param: hl.Param):
    if param.type() == hl.Int(32):
        return f"int {param.name()}"
    elif param.type() == hl.UInt(8):
        return f"uint8_t {param.name()}"
    elif param.type() == hl.Float(32):
        return f"float {param.name()}"
    else:
        raise RuntimeError(f"Param has unknown type: {param}")


def _halide_param_to_cpp_variable(param: hl.Param):
    return f"{param.name()}"


import halide as hl
import numpy as np


def lerp(a, b, t):
    return a * (1 - t) + b * t


def _viridis_colormap(value: hl.Func):
    viridis_data = np.array(
        [
            [0.267004, 0.004874, 0.329415],
            [0.283148, 0.130895, 0.449241],
            [0.253935, 0.265254, 0.529983],
            [0.163625, 0.471133, 0.558148],
            [0.134692, 0.658636, 0.517649],
            [0.477504, 0.821444, 0.318195],
            [0.993248, 0.906157, 0.143936],
        ]
    )
    viridis_buffer = hl.Buffer(viridis_data.transpose())
    viridis = hl.BoundaryConditions.repeat_edge(viridis_buffer)

    x = hl.Var("x")
    y = hl.Var("y")
    c = hl.Var("c")

    idx = value[x, y] * (len(viridis_data) - 1)
    lower_idx = hl.i32(hl.floor(idx))
    upper_idx = hl.i32(hl.ceil(idx))

    lower_color = hl.Func("lower_color")
    lower_color[x, y, c] = viridis[lower_idx, c]
    upper_color = hl.Func("upper_color")
    upper_color[x, y, c] = viridis[upper_idx, c]
    t = idx - lower_idx

    return lerp(lower_color[x, y, c], upper_color[x, y, c], t)


def interact(
    func: hl.Func,
    size: List[int],
    controls: Optional[List[Slider]] = None,
    output_path: Optional[Path] = None,
):
    embed_javascript = output_path is None
    if output_path is not None:
        if not output_path.exists():
            output_path.mkdir(parents=True)
        else:
            assert output_path.is_dir(), "`output_path` must be a directory"

    controls = controls if controls else []

    func_name_original = func.name().replace("$", "")
    unique_id = uuid.uuid4().hex
    func_name = f"{func_name_original}_{unique_id}"

    width = size[0]
    height = size[1]

    assert (
        len(size) == func.dimensions()
    ), f"The number of dimensions in the Func (`{func.dimensions()} does not match the size requested ({len(size)})."

    x = hl.Var("x")
    y = hl.Var("y")
    c = hl.Var("c")

    match func.dimensions():
        case 2:
            rdom = hl.RDom([(0, size[0]), (0, size[1])])
            min_value = hl.Func("min_value")
            min_value[()] = hl.minimum(hl.f32(func[rdom.x, rdom.y]))
            max_value = hl.Func("max_value")
            max_value[()] = hl.maximum(hl.f32(func[rdom.x, rdom.y]))
            min_value.compute_root()
            max_value.compute_root()
            normalized = hl.Func("normalized")
            normalized[x, y] = (hl.f32(func[x, y]) - min_value[()]) / (max_value[()] - min_value[()])
            wrapper = hl.Func(f"{func_name}_wrapper")
            wrapper[x, y, c] = hl.cast(hl.UInt(8), _viridis_colormap(normalized) * 255)
            wrapper[x, y, 3] = hl.u8(255)
        case 3:
            wrapper = hl.Func(f"{func_name}_wrapper")
            if (func.type() == hl.Float(32)) | (func.type() == hl.Float(64)):
                wrapper[x, y, c] = hl.cast(hl.UInt(8), func[x, y, hl.clamp(c, 0, 2)] * 255.0)
            else:
                wrapper[x, y, c] = hl.cast(hl.UInt(8), func[x, y, hl.clamp(c, 0, 2)])
            wrapper[x, y, 3] = hl.u8(255)
        case dims:
            raise NotImplementedError(f"Func objects of {dims} dimensions are currently not supported")

    target = hl.Target("wasm-32-wasmrt")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        include_path = Path(__file__).parent.parent.parent / "include"
        js_path = temp_dir / f"{func_name}.js"
        wasm_path = temp_dir / f"{func_name}.wasm"
        wrapper_path = temp_dir / f"{func_name}_wrapper.cpp"
        object_path = temp_dir / f"{func_name}.o"
        header_path = temp_dir / f"{func_name}.h"
        wrapper.output_buffer().dim(0).set_stride(4)
        wrapper.output_buffer().dim(2).set_stride(1)

        all_params = [control.param for control in controls]
        wrapper.compile_to(
            {
                hl.OutputFileType.object: str(object_path),
                hl.OutputFileType.c_header: str(header_path),
            },
            arguments=all_params,
            fn_name=func_name,
            target=target,
        )

        def maybe_append_comma(string: str):
            return string + ", " if len(string) != 0 else ""

        cpp_arguments = maybe_append_comma(
            ", ".join([_halide_param_to_cpp_argument(control.param) for control in controls])
        )
        cpp_variables = maybe_append_comma(
            ", ".join([_halide_param_to_cpp_variable(control.param) for control in controls])
        )

        wrapper_code = f"""
    #include "HalideBuffer.h"
    #include "{func_name}.h"
    #include <emscripten.h>
    #include <emscripten/bind.h>
    #include <emscripten/val.h>
    #include <iostream>

    class Uint8Buffer {{
    public:
        Uint8Buffer(int width, int height, int channels) :
            buffer(Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, channels))
            {{}}
        int size_in_bytes() {{ return buffer.size_in_bytes(); }};
        int data() {{ return (int)(buffer.data()); }};

        Halide::Runtime::Buffer<uint8_t> buffer;
    }};

    int process_image(
        {cpp_arguments}
        int width,
        int height,
        Uint8Buffer output_buffer
        ) {{
      const auto result = {func_name}(
        {cpp_variables}
        output_buffer.buffer
      );
      return result;
    }}

    EMSCRIPTEN_BINDINGS(my_module) {{
      emscripten::function("process_image", &process_image);

      emscripten::class_<Uint8Buffer>("Uint8Buffer")
        .constructor<int, int, int>()
        .function("data", &Uint8Buffer::data)
        .function("size_in_bytes", &Uint8Buffer::size_in_bytes);
    }}
    """
        wrapper_path.write_text(wrapper_code)

        notebook_args = ["-sSINGLE_FILE"] if embed_javascript else []

        subprocess.call(
            [
                "em++",
                "-lembind",
                str(wrapper_path),
                str(object_path),
                "-sMODULARIZE",
                f"-sEXPORT_NAME=Module_{func_name}",
                "-I",
                include_path,
                "-std=c++17",
                "-s",
                "WASM=1",
                "-sTOTAL_MEMORY=65536000",
                "-sASSERTIONS",
                "-sALLOW_MEMORY_GROWTH",
                *notebook_args,
                "-O3",
                "-o",
                str(js_path),
            ]
        )

        sliders = "<br>".join(
            f"""
            {control.param.name()}:
            <span id="{func_name}_{control.param.name()}_slider_value">{control.initial_value}</span>
            <br>
            <input
                type="range"
                min="{control.min_value}"
                max="{control.max_value}"
                step="{control.step}"
                value="{control.initial_value}"
                class="slider"
                id="{func_name}_{control.param.name()}_slider"
            >
            """
            for control in controls
        )

        slider_updates = "".join(
            f"""
            document.getElementById("{func_name}_{control.param.name()}_slider").oninput = (event) => {{
                document.getElementById("{func_name}_{control.param.name()}_slider_value").innerText = event.target.value;
                update({width}, {height});
            }};
            """
            for control in controls
        )

        slider_arguments = maybe_append_comma(
            ", ".join(
                f"""parseFloat(document.getElementById("{func_name}_{control.param.name()}_slider").value)"""
                for control in controls
            )
        )
        initialize_sliders = "\n".join(
            f"""document.getElementById("{func_name}_{control.param.name()}_slider").value = {control.initial_value};"""
            for control in controls
        )

        html = f"""
        <canvas id="canvas" style="display: none;"></canvas>
        <canvas id="inputcanvas" style="display: none;"></canvas>
        <canvas id="{func_name}_canvas" width=256 height=256 style="width: 100%"></canvas>
        <br>
        {sliders}
        <script>
            const run_{func_name} = async () => {{
                {initialize_sliders}
                const {func_name} = await Module_{func_name}();
                const canvas = document.getElementById('inputcanvas');
                const context = canvas.getContext('2d');
                const img = new Image();
                const update = (width, height) => {{
                    const output = new {func_name}.Uint8Buffer(width, height, 4);
                    const result = {func_name}.process_image({slider_arguments} width, height, output);
                    const clamped = new Uint8ClampedArray({func_name}.HEAP8.buffer, output.data(), output.size_in_bytes());
                    const imageData = new ImageData(clamped, width, height);
                    const canvas = document.getElementById("{func_name}_canvas");
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext("2d");
                    ctx.putImageData(imageData, 0, 0);
                }};
                const updateSlider = () => {{
                    update({width}, {height});
                }};
                {slider_updates}
                update({width}, {height});
            }}
        </script>
        """
        if embed_javascript:
            contents = js_path.read_text(encoding="utf-8")
            html += f"""
            <script>{contents}</script>
            <script>run_{func_name}();</script>
            """
        else:
            shutil.copy(js_path, output_path)
            shutil.copy(wasm_path, output_path)
            html += f"""
            <script src="/{func_name}.js" onload="run_{func_name}()"></script>
            """
    display(HTML(html))
