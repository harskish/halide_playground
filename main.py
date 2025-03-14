import os
import platform
from pathlib import Path
import subprocess
from types import SimpleNamespace
import halide as hl
import numpy as np
from PIL import Image
from glfw import KEY_F5
from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from halide_ffi_ctypes import LibBinder, Buffer

# Halide C++ library bundled with python package
hl_root = Path(hl.install_dir())
hl_header = hl_root / 'include' / 'HalideRuntime.h'
versions = [l for l in hl_header.read_text().splitlines() if l.startswith('#define HALIDE_VERSION_')]
major, minor, patch = [int(v.split()[-1]) for v in versions]
assert (major, minor, patch) == (19, 0, 0)
assert platform.system() == 'Windows', 'Not yet tested on other platforms'

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
    kernel: Param = EnumParam('Kernel', 'blur', ['write', 'blur'])
    out_WH: Param = Int2Param('Output (W, H)', (1024, 681), 32, 4096)
    
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

        self.prev_kernel_contents = ''
        ret = self.recompile_and_run()
        self.update_image(ret.numpy_hwc())
    
    def recompile(self):
        filepath = Path(f'examples/{self.state.kernel}.cpp')

        contents = filepath.read_text()
        if contents == self.prev_kernel_contents:
            print('Contents identical!')
            return

        stem = filepath.stem
        basename = Path('.') / 'build' / stem # relative to examples/
        libname_rel = basename.with_suffix('.dll')
        libname_abs = Path(__file__).parent / 'examples' / libname_rel.as_posix()

        os.environ['STEM'] = stem
        os.environ['HALIDE_PATH'] = str(hl_root.resolve()) # Windows path syntax

        # Windows: close DLL handle (otherwise compiler can't overwrite)
        self.binder.close()

        #print('State pre:', libname_abs.stat())

        lib_mtime = libname_abs.stat().st_mtime # modification
        os.makedirs('examples/build/tmp', exist_ok=True)
        run_in_vs2022_cmd('nmake', '/f', 'NMakefile', str(libname_rel), cwd='examples')
        
        #print('State post:', libname_abs.stat())

        if libname_abs.stat().st_mtime == lib_mtime and self.prev_kernel_contents != '':
            print('Library unchanged, compilation probably failed')
        else:
            self.prev_kernel_contents = contents

        self.binder.prepare(*self.state.out_WH, self.out_ch)
        self.vars = self.binder.bind("render", libname_abs.as_posix(), self.args)
        self.setup_widgets()
    
    def recompile_and_run(self):
        self.recompile()
        return self.run_pipeline()
    
    def setup_widgets(self):
        new_params = {}
        for v in self.vars:
            label = f'{v.name}##{self.state.kernel}'
            if v.buffer:
                H, W, C = self.input_img.shape
                buff: Buffer = v.make_buffer(W, H, C)
                buff.from_numpy(self.input_img)
                # TODO: ConstantParam?
                new_params[v.name] = SimpleNamespace(value=buff.buffer) # shared across kernels
            elif v.type == 'float':
                new_params[label] = FloatParam(label, v.default, v.min, v.max)
            elif v.type == 'int':
                new_params[label] = IntParam(label, v.default, v.min, v.max)
            else:
                raise ValueError(f"Unhandled variable: {v}")
    
        # Carry values over
        # Label contains kernel name => unique widgets
        for k in new_params:
            if k in self.params:
                new_params[k] = self.params[k]
        
        # Atomically replace
        self.params = new_params
        
    def run_pipeline(self):
        self.args.clear() # haldle kept by binder, cannot replace
        for k, meta in zip(self.params, self.vars):
            name = k.split('##')[0]
            assert meta.name == name
            value = self.params[k].value
            self.args[name] = meta.cast_fun(value) if hasattr(meta, 'cast_fun') else value
        
        outbuf, _ = self.binder.call() # TODO: new args as input?
        return outbuf
    
    def draw_toolbar(self):
        for k, v in self.params.items():
            if isinstance(v, Param):
                v.draw()
    
    def draw_pre(self):
        pass # fullscreen plotting (examples/plotting.py)

    def draw_overlays(self, draw_list):
        pass # draw on top of UI elements

    def draw_output_extra(self):
        pass # draw below main output (see `pad_bottom` in constructor)

    def draw_menu(self):
        pass

    def drag_and_drop_callback(self, paths: list[Path]):
        pass

    def compute(self):
        if self.v.keyhit(KEY_F5):
            buff = self.recompile_and_run()
            return buff.numpy_hwc()
        return None # reuse cached

if __name__ == '__main__':
    viewer = Viewer('Halide Playground')
