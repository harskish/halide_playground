import os
import platform
from pathlib import Path
import subprocess
import halide as hl
import numpy as np
from halide_ffi_ctypes import LibBinder, Buffer
from pyviewer import single_image_viewer as siv
siv.init('siv', sync=False, hidden=True)

def run_in_vs2022_cmd(*args, cwd=None, shell=False):
    # If vcvars already set, don't re-run
    if 'VisualStudioVersion' in os.environ:
        print('vcvars already set')
        return subprocess.run([*args], cwd=cwd, shell=shell)
    
    vs_path = "C:/Program Files/Microsoft Visual Studio/2022/Community"
    vcvarsall_path = f"{vs_path}/VC/Auxiliary/Build/vcvars64.bat"
    assert os.path.exists(vs_path), f'Could not find Visual Studio install at "{vs_path}"'
    assert os.path.exists(vcvarsall_path), 'Could not find vcvars64.bat'
    return subprocess.run([vcvarsall_path, '&&', *args], cwd=cwd, shell=shell)

assert platform.system() == 'Windows'

filepath = Path('examples/blur.cpp')
rootdir = filepath.parent
ext = filepath.suffix
stem = filepath.stem

basename = Path('.') / 'build' / stem # relative to examples/
libname = basename.with_suffix('.dll')

# Halide C++ library bundled with python package
hl_root = Path(hl.install_dir())
hl_header = hl_root / 'include' / 'HalideRuntime.h'
versions = [l for l in hl_header.read_text().splitlines() if l.startswith('#define HALIDE_VERSION_')]
major, minor, patch = [int(v.split()[-1]) for v in versions]
assert (major, minor, patch) == (19, 0, 0)

os.environ['STEM'] = stem
os.environ['HALIDE_PATH'] = str(hl_root.resolve()) # Windows path syntax

os.makedirs('examples/build/tmp', exist_ok=True)

#print('Skipping build')
run_in_vs2022_cmd('nmake', '/f', 'NMakefile', str(libname), cwd='examples')

# observeSaves():                       # file:///./lib/atomic-halide.coffee#L37
# => onDidSave() => refresh()           # file:///./lib/atomic-halide.coffee#L43
# refresh() => listParams():            # file:///./lib//atomic-halide-view.coffee#L383
# makeRenderer() => returns asdf.dll
# rendererBinding.bind()                # file:///./lib/atomic-halide-view.coffee#L399
# bind:                                 # file:///./lib/halide-lib-binder.coffee#L271

binder = LibBinder()
libname_abs = Path(__file__).parent / 'examples' / libname.as_posix()
assert libname_abs.is_file()

siv.inst.show()

from PIL import Image
input_img = np.array(Image.open('examples/ueno.jpg').convert('RGB'))
H, W, C = input_img.shape

for radius in [1, 5, 11, 21]:
    args = {} # handle passed to bind
    binder.close()
    binder.prepare(W, H, C)
    vars = binder.bind("render", libname_abs.as_posix(), args)

    for v in vars:
        name = v['name']
        if v.get('buffer', False):
            buff: Buffer = v['make_buffer'](W, H, C)
            buff.from_numpy(input_img)
            args[v["name"]] = buff.buffer
        elif v.get('int', False):
            args[v["name"]] = v["default"]
        else:
            raise ValueError(f"Unhandled variable: {v}")
    
    args['radius'] = radius
    
    outbuf, errors = binder.call()
    siv.draw(img_hwc=outbuf.numpy_hwc())

print('Done')