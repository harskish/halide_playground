from collections import defaultdict
from functools import lru_cache
import os
import platform
from pathlib import Path
import subprocess
from types import SimpleNamespace
import halide as hl
import numpy as np
import time
from PIL import Image
from glfw import KEY_LEFT_SUPER, KEY_LEFT_CONTROL, KEY_S
from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from halide_ffi_ctypes import LibBinder, Buffer
import rawpy
import ctypes

from pyviewer.imgui_themes import color_uint as hex_color
from imgui_bundle import imgui, imgui_md # type: ignore
from imgui_bundle import imgui_color_text_edit as ed # type: ignore

# Halide C++ library bundled with python package
hl_root = Path(hl.install_dir())
hl_header = hl_root / 'include' / 'HalideRuntime.h'
versions = [l for l in hl_header.read_text().splitlines() if l.startswith('#define HALIDE_VERSION_')]
major, minor, patch = [int(v.split()[-1]) for v in versions]
assert (major, minor, patch) == (19, 0, 0)

def run_in_vs2022_cmd(*args, cwd=None, shell=False):
    if 'VisualStudioVersion' in os.environ:
        print('vcvars already set')
        return subprocess.run([*args], cwd=cwd, shell=shell)
    
    vs_path = "C:/Program Files/Microsoft Visual Studio/2022/Community"
    vcvarsall_path = f"{vs_path}/VC/Auxiliary/Build/vcvars64.bat"
    assert os.path.exists(vs_path), f'Could not find Visual Studio install at "{vs_path}"'
    assert os.path.exists(vcvarsall_path), 'Could not find vcvars64.bat'
    return subprocess.run([vcvarsall_path, '&&', *args], cwd=cwd, shell=shell)

@strict_dataclass
class CommonState(ParamContainer):
    kernel: Param = EnumParam('Kernel', 'fuji_debayer', ['write', 'blur', 'fuji_debayer'])
    out_WH: Param = Int2Param('Output (W, H)', (6032, 4028), 32, 8000)
    input: Param = EnumSliderParam('Input', 'Ueno', ['Ueno', 'Black'])
    
class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state = CommonState()
        self.state_last = None
        self.cache = {}
        self.input_img = np.array(Image.open('examples/ueno.jpg').convert('RGB'))
        self.out_ch = self.input_img.shape[-1]
        
        # UI widgets created dynamically based on HL pipeline metadata
        self.params: dict[str, Param] = {} # label -> Param
        self.binder = LibBinder()
        self.vars = None # pipeline metadata
        self.args = {} # pipeline input arguments

        # Initialize code editor
        self.editor_active_kernel = self.state.kernel
        self.kernels_dir = Path(__file__).parent / 'examples'
        self.editor_sources = { k: (self.kernels_dir / f'{k}.cpp').read_text() for k in self.state['kernel'].opts }
        self.editor = ed.TextEditor()
        self.editor.set_language_definition(ed.TextEditor.LanguageDefinition.c_plus_plus())
        self.editor.set_text(self.editor_sources[self.state.kernel])        
        self.editor.set_show_whitespaces(False)
        palette = self.editor.get_dark_palette()
        palette[ed.TextEditor.PaletteIndex.number] = imgui.IM_COL32(*hex_color("#CAC074")[::-1]) # endianness mismatch?
        self.editor.set_palette(palette)

        self.prev_kernel_contents = defaultdict(str)
        ret = self.recompile_and_run()
        self.update_image(ret.numpy_hwc())
    
    # Override to be extra wide
    @property
    def toolbar_width(self):
        return int(round(600 * self.v.ui_scale))
    
    def recompile(self):
        self.set_window_title('Recompiling...')
        filepath = Path(f'examples/{self.state.kernel}.cpp')
        stem = filepath.stem
        basename = Path('.') / 'build' / stem # relative to examples/
        lib_ext = { 'Windows': '.dll', 'Linux': '.so', 'Darwin': '.dylib' }[platform.system()]
        libname_rel = basename.with_suffix(lib_ext)
        libname_abs = Path(__file__).parent / 'examples' / libname_rel.as_posix()

        # Close library handle before reloading
        self.binder.close()

        # Only recompile if sources have changed
        contents = filepath.read_text()
        if contents != self.prev_kernel_contents[self.state.kernel]:
            os.environ['STEM'] = stem
            os.environ['HALIDE_PATH'] = str(hl_root.resolve()) # Windows path syntax

            lib_mtime = libname_abs.stat().st_mtime if libname_abs.is_file() else 0 # modification
            os.makedirs('examples/build/tmp', exist_ok=True)
            
            if platform.system() == 'Windows':
                run_in_vs2022_cmd('nmake', '/f', 'NMakefile', str(libname_rel), cwd='examples')
            else:
                subprocess.run(['make', str(libname_rel)], cwd='examples')

            if libname_abs.stat().st_mtime == lib_mtime and self.prev_kernel_contents[self.state.kernel] != '':
                print('Library unchanged, compilation probably failed')
            else:
                self.prev_kernel_contents[self.state.kernel] = contents
        else:
            pass #print('Contents identical!')
        
        self.vars, out_dtype = self.binder.bind("render", libname_abs.as_posix(), self.args)
        self.binder.prepare(*self.state.out_WH, self.out_ch, out_dtype)
        self.setup_widgets()
    
    def recompile_and_run(self):
        self.recompile()
        return self.run_pipeline()
    
    def buffers_fuji_debayer(self):
        raw = rawpy.imread('C:/Users/Erik/code/isp/data/20240804_144851.RAF') # HxW
        rotate = lambda arr: np.rot90(arr, k={3: 2, 5: 1, 6: 3}.get(raw.sizes.flip, 0))
        return {
            'cfa': rotate(raw.raw_image_visible)[..., None].copy(), # as HWC
            'colors': rotate(raw.raw_colors_visible)[..., None].copy(),
            'color_desc': raw.color_desc.decode(), # RGBG
            'pattern': raw.raw_pattern.copy(), # 6x6
            'white_level': raw.white_level,
            'black_level': raw.black_level_per_channel[0],
            'camera_whitebalance_r': raw.camera_whitebalance[0],
            'camera_whitebalance_g': raw.camera_whitebalance[1],
            'camera_whitebalance_b': raw.camera_whitebalance[2],
            'sizes': raw.sizes._asdict(),
        }
    
    @lru_cache
    def get_buffers(self, kernel, input):
        """Get dict of input buffers required for pipeline"""
        if kernel == 'fuji_debayer':
            return self.buffers_fuji_debayer()
        if input == 'Ueno':
            return {'image': self.input_img}
        elif input == 'Black':
            return { 'image': np.zeros_like(self.input_img) }
        else:
            raise ValueError(f"Unknown input: {input}")
    
    def setup_widgets(self):
        buffers = self.get_buffers(self.state.kernel, self.state.input)
        
        new_params = {}
        for v in self.vars:
            label = f'{v.name}##{self.state.kernel}'
            if v.buffer:
                input = buffers[v.name]
                H, W, C = input.shape
                buff: Buffer = v.make_buffer(W, H, C, dtype=np.ctypeslib.as_ctypes_type(input.dtype))
                buff.from_numpy(input)
                new_params[label] = SimpleNamespace(value=buff.buffer) # shared across kernels. TODO: ConstantParam?
            elif v.name in buffers:
                # Params with constant values, don't need sliders
                new_params[label] = SimpleNamespace(value=buffers[v.name])
            elif v.type == 'float':
                new_params[label] = FloatParam(label, v.default, v.min, v.max)
            elif v.type == 'int':
                new_params[label] = IntParam(label, v.default, v.min, v.max)
            elif v.type == 'uint':
                new_params[label] = IntParam(label, v.default, max(0, v.min), v.max)
            else:
                raise ValueError(f"Unhandled variable: {v}")
    
        # Carry values over
        # Label contains kernel name => unique widgets
        for k in new_params:
           if k in self.params and isinstance(new_params[k], Param):
               p = new_params[k]
               p.value = max(p.min, min(self.params[k].value, p.max)) # copy value, update ranges
        
        # Atomically replace
        self.params = new_params
    
    def status(self, msg, end='\n'):
        print(msg, end=end)
        self.set_window_title(msg.strip())

    def run_pipeline(self):
        self.args.clear() # haldle kept by binder, cannot replace
        for k, meta in zip(self.params, self.vars):
            name = k.split('##')[0]
            assert meta.name == name
            value = self.params[k].value
            self.args[name] = meta.cast_fun(value) if hasattr(meta, 'cast_fun') else value
        
        t0 = time.perf_counter()
        self.status(f'{self.state.kernel}: ...', end='')
        outbuf, error_str = self.binder.call() # TODO: new args as input?
        dt = time.perf_counter() - t0
        time_str = f'{dt:.2f}s' if dt > 1 else f'{dt*1000:.1f}ms'
        suffix = f'Error - {error_str}' if error_str else time_str
        self.status(f'\r{self.state.kernel}: {suffix}')

        return outbuf

    def editor_save_action(self):
        """Editor save action, triggered by ctrl+s or cmd+s."""
        if self.state.kernel != self.editor_active_kernel:
            print('Mid-switch, aborting')
            return
        
        kernel_name = self.state.kernel
        pth = self.kernels_dir / f'{kernel_name}.cpp'
        assert pth.is_file(), f"File not found: {pth}"
        new_src = self.editor.get_text()
        if self.state.kernel != kernel_name:
            print('Kernel selection changed during save, aborting')
            return
        pth.write_text(new_src)

    def draw_toolbar(self):
        for k, v in self.params.items():
            if isinstance(v, Param):
                v.draw()

        # Detect source file change
        if self.state.kernel != self.editor_active_kernel:
            self.editor_sources[self.editor_active_kernel] = self.editor.get_text() # write updated back
            self.editor.set_text(self.editor_sources[self.state.kernel])
            self.editor_active_kernel = self.state.kernel
        
        self.editor.render(f"{self.state.kernel}.cpp")

    def compute(self):
        # imgui_bundle version seems broken:
        # https://github.com/pthom/ImGuiColorTextEdit/blob/165ca5fe8be900884c88b90f16955bbf848b23ee/TextEditor.cpp#L2027
        # https://github.com/BalazsJako/ImGuiColorTextEdit/blob/0a88824f7de8d0bd11d8419066caa7d3469395c4/TextEditor.cpp#L702C17-L702C38
        # imgui.get_io().config_mac_osx_behaviors = True
        mod_key = KEY_LEFT_SUPER if platform.system() == 'Darwin' else KEY_LEFT_CONTROL
        if self.v.keyhit(KEY_S) and self.v.keydown(mod_key): # S first to clear state
            self.editor_save_action()
            buff = self.recompile_and_run()
            return buff.numpy_hwc() if buff is not None else None
        return None # reuse cached

if __name__ == '__main__':
    viewer = Viewer('Halide Playground')


# Links
# https://github.com/bobobo1618/halideraw
# https://github.com/zshipko/halide-runtime
# https://github.com/dragly/numlide
# https://github.com/dragly/halide-widgets (https://github.com/dragly?tab=repositories)
# https://github.com/anuejn/halide_experiments/blob/main/halide_blocks/debayer.py