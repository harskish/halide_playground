from collections import defaultdict
from contextlib import contextmanager, nullcontext
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
from pyviewer.params import *
from halide_ffi_ctypes import LibBinder, Buffer
import rawpy
from pyviewer.gl_viewer import _texture
import OpenGL.GL as gl
import threading
import glfw

import json
from enum import Enum
import time
from imgui_bundle import hello_imgui, imgui, ImVec4 # type: ignore
from imgui_bundle.demos_python.demo_utils import demos_assets_folder
from typing import List
import threading

from pyviewer.toolbar_viewer import PannableArea
from pyviewer.utils import normalize_image_data
from pyviewer.imgui_themes import color_uint as hex_color
from imgui_bundle import imgui, imgui_md # type: ignore
from imgui_bundle import imgui_color_text_edit as ed # type: ignore
from imgui_bundle import immapp # thin wrapper around hello_imgui that also instantiates addons

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
    return subprocess.run([vcvarsall_path, '&&', *args], cwd=cwd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@strict_dataclass
class AppState:
    f: float = 0
    counter: int = 0
    title_font: imgui.ImFont = None
    color_font: imgui.ImFont = None
    emoji_font: imgui.ImFont = None
    large_icon_font: imgui.ImFont = None
    start_event: threading.Event = threading.Event()
    stop_event: threading.Event = threading.Event()
    image: np.ndarray = None
    img_dt: float = 0
    last_upload_dt: float = 0
    tex_handle: _texture = None # created after GL init

@strict_dataclass
class CommonState(ParamContainer):
    kernel: Param = EnumParam('Kernel', 'fuji_debayer', ['write', 'blur', 'fuji_debayer'])
    out_WH: Param = Int2Param('Output (W, H)', (6032, 4028), 32, 8000)
    input: Param = EnumSliderParam('Input', 'Ueno', ['Ueno', 'Black'])

class Viewer:
    def __init__(self):
        # Installed by pip, includes two fonts
        hello_imgui.set_assets_folder(demos_assets_folder())

        # Our application state
        app_state = AppState()
        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = "Halide Playground"
        runner_params.imgui_window_params.menu_app_title = "HLP"
        runner_params.app_window_params.window_geometry.size = (1000, 900)
        runner_params.app_window_params.restore_previous_geometry = True

        # Normally setting no_mouse_input windows flags on containing window is enough,
        # but docking (presumably) seems to be capturing mouse input regardless.
        self.pan_handler = PannableArea(force_mouse_capture=True)
        
        def post_init_fun():
            self.setup_state()
            app_state.start_event.set()

        def before_exit():
            del self.app_state.tex_handle
            app_state.stop_event.set()
        
        def add_backend_cbk(*args, **kwargs):
            # Set own glfw callbacks, will be chained by imgui
            window = glfw.get_current_context()
            self.pan_handler.set_callbacks(window)

        runner_params.callbacks.post_init = post_init_fun
        runner_params.callbacks.before_exit = before_exit
        runner_params.callbacks.post_init_add_platform_backend_callbacks = add_backend_cbk

        self.app_state = app_state

        # State exists, start compute thread asap
        compute_thread = threading.Thread(target=self.compute_loop, args=[], daemon=True)
        compute_thread.start()

        # HDR output (only available on Metal)
        # TODO: probably need to set metal backend first
        # TODO: do in callbacks.post_init_add_platform_backend_callbacks?
        if hello_imgui.has_edr_support():
            # https://github.com/pthom/hello_imgui/blob/51d850fc/src/hello_imgui_demos/hello_edr/hello_edr.mm
            runner_params.renderer_backend_type = hello_imgui.RendererBackendType.metal
            renderer_backend_options = hello_imgui.RendererBackendOptions()
            renderer_backend_options.request_float_buffer = True
        
        # Fonts
        runner_params.callbacks.load_additional_fonts = lambda: load_fonts(app_state)
        
        # Status bar: fps etc.
        runner_params.imgui_window_params.show_status_bar = True
        #runner_params.callbacks.show_status = lambda: status_bar_gui(app_state)

        # Change style
        #runner_params.callbacks.setup_imgui_style = setup_my_theme

        # Create "MainDockSpace"
        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        )

        # Allow splitting into separate windows?
        runner_params.imgui_window_params.enable_viewports = True
        
        # Docking layout
        runner_params.docking_params = hello_imgui.DockingParams()
        runner_params.docking_params.docking_splits = create_default_docking_splits()

        # Text editor: left-hand side
        text_editor_window = hello_imgui.DockableWindow(can_be_closed_=False)
        text_editor_window.label = "Text editor"
        text_editor_window.dock_space_name = "CommandSpace"
        text_editor_window.gui_function = self.draw_text_editor

        # Output image: right top
        output_window = hello_imgui.DockableWindow(can_be_closed_=False)
        output_window.label = "Pipeline Output"
        output_window.dock_space_name = "MainDockSpace"
        output_window.gui_function = self.draw_output

        # Param sliders: right bottom
        widgets_window = hello_imgui.DockableWindow(can_be_closed_=False)
        widgets_window.label = "Input Params"
        widgets_window.dock_space_name = "CommandSpace2"
        widgets_window.gui_function = self.draw_toolbar

        dockable_windows = [
            text_editor_window,
            output_window,
            widgets_window,
        ]
        
        runner_params.docking_params.dockable_windows = dockable_windows
        runner_params.docking_params.main_dock_space_node_flags |= imgui.DockNodeFlags_.auto_hide_tab_bar

        # .ini for window and app state saving
        runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        runner_params.ini_filename = 'halide_playground.ini'
        ini_path = os.path.join(hello_imgui.ini_folder_location(runner_params.ini_folder_type), runner_params.ini_filename)
        print(f'INI path: {ini_path}')

        addons = immapp.AddOnsParams(with_markdown=True)
        immapp.run(runner_params, add_ons_params=addons)

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

        self.app_state.tex_handle = _texture(gl.GL_NEAREST, gl.GL_NEAREST)
        self.prev_kernel_contents = defaultdict(str)
        self.should_recompile = True # set by UI thread, reacted to by compute thread
    
    def update_image(self, arr):
        assert isinstance(arr, np.ndarray)
        
        # Eventually uploaded by UI thread
        self.app_state.image = normalize_image_data(arr, 'uint8')
        self.app_state.img_dt = time.monotonic()

    def draw_output(self):
        # Need to do upload from main thread
        if self.app_state.img_dt > self.app_state.last_upload_dt:
            self.app_state.tex_handle.upload_np(self.app_state.image)
            self.app_state.last_upload_dt = time.monotonic()
        
        # Reallocate if window size has changed
        if self.app_state.image is not None:
            tH, tW, _ = self.app_state.image.shape
            cW, cH = map(int, imgui.get_content_region_avail())
            canvas_tex = self.pan_handler.draw_to_canvas(self.app_state.tex_handle.tex, tW, tH, cW, cH)
            imgui.image(canvas_tex, (cW, cH))

    def set_window_title(self, title):
        pass
    
    def recompile(self) -> str:
        """Recompile the current kernel, return potential error message"""
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
            print('Compiling...')
            
            os.environ['STEM'] = stem
            os.environ['HALIDE_PATH'] = str(hl_root.resolve()) # Windows path syntax

            lib_mtime = libname_abs.stat().st_mtime if libname_abs.is_file() else 0 # modification
            os.makedirs('examples/build/tmp', exist_ok=True)
            
            if platform.system() == 'Windows':
                if len(os.environ['PATH']) > 7_000:
                    print('Warning: PATH very long, nmake might fail')
                ret = run_in_vs2022_cmd('nmake', '/f', 'NMakefile', str(libname_rel), cwd='examples')
                print('STDOUT:', ret.stdout.decode())
                if ret.returncode != 0:
                    err_msg = ret.stderr.decode().strip()
                    #print(f'STDERR:\n"{err_msg}"')
                    return err_msg
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
        return '' # no error
    
    def recompile_and_run(self):
        error_msg = self.recompile()
        if error_msg:
            msg_single_line = error_msg.replace('\n', ' ').replace('\r', '')
            self.status(f'\r{self.state.kernel}: {msg_single_line}')
            return None
        return self.run_pipeline()
    
    def buffers_fuji_debayer(self):
        raw = rawpy.imread('/Users/Erik/code/isp/data/20240804_144851.RAF') # HxW
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

        return outbuf.numpy_hwc() if outbuf is not None else None

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

    def draw_text_editor(self):
        # Detect source file change
        if self.state.kernel != self.editor_active_kernel:
            self.editor_sources[self.editor_active_kernel] = self.editor.get_text() # write updated back
            self.editor.set_text(self.editor_sources[self.state.kernel])
            self.editor_active_kernel = self.state.kernel

        # Detect save action
        mod_key = imgui.Key.left_ctrl # actually means cmd on macOS...?!
        if imgui.is_key_pressed(imgui.Key.s) and imgui.is_key_down(mod_key):
            self.editor_save_action()
            self.should_recompile = True
        
        # imgui_bundle version seems broken:
        # https://github.com/pthom/ImGuiColorTextEdit/blob/165ca5fe8be900884c88b90f16955bbf848b23ee/TextEditor.cpp#L2027
        # https://github.com/BalazsJako/ImGuiColorTextEdit/blob/0a88824f7de8d0bd11d8419066caa7d3469395c4/TextEditor.cpp#L702C17-L702C38
        # imgui.get_io().config_mac_osx_behaviors = True
        imgui.push_font(imgui_md.get_code_font())
        self.editor.render(f"{self.state.kernel}.cpp")
        imgui.pop_font()
    
    def draw_toolbar(self):
        draw_container(self.state)
        for k, v in self.params.items():
            if isinstance(v, Param):
                v.draw()

    def compute_loop(self):
        print('Compute thread: waiting for start event')
        self.app_state.start_event.wait(timeout=None)
        print('Compute thread: start event received')

        while not self.app_state.stop_event.is_set():
            self.compute()
            time.sleep(0.01)
        
        print('Compute thread: received stop event, exiting')

    def compute(self):
        if self.should_recompile:
            ret = self.recompile_and_run()
            if ret is not None:
                self.update_image(ret)
            self.should_recompile = False

        return None # reuse cached

# Links
# https://github.com/bobobo1618/halideraw
# https://github.com/zshipko/halide-runtime
# https://github.com/dragly/numlide
# https://github.com/dragly/halide-widgets (https://github.com/dragly?tab=repositories)
# https://github.com/anuejn/halide_experiments/blob/main/halide_blocks/debayer.py


def load_fonts(app_state: AppState):  # This is called by runnerParams.callbacks.LoadAdditionalFonts
    # First, load the default font (the default font should be loaded first)
    # In this example, we instruct HelloImGui to use FontAwesome6 instead of FontAwesome4
    hello_imgui.get_runner_params().callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
    hello_imgui.imgui_default_settings.load_default_font_with_font_awesome_icons()

    # Load the title font
    app_state.title_font = hello_imgui.load_font("fonts/DroidSans.ttf", 18.0)
    font_loading_params_title_icons = hello_imgui.FontLoadingParams()
    font_loading_params_title_icons.merge_to_last_font = True
    font_loading_params_title_icons.use_full_glyph_range = True
    app_state.title_font = hello_imgui.load_font("fonts/Font_Awesome_6_Free-Solid-900.otf",
                                                 18.0, font_loading_params_title_icons)

    # Load the emoji font
    font_loading_params_emoji = hello_imgui.FontLoadingParams()
    font_loading_params_emoji.use_full_glyph_range = True
    app_state.emoji_font = hello_imgui.load_font("fonts/NotoEmoji-Regular.ttf", 24., font_loading_params_emoji)

    # Load a large icon font
    font_loading_params_large_icon = hello_imgui.FontLoadingParams()
    font_loading_params_large_icon.use_full_glyph_range = True
    app_state.large_icon_font = hello_imgui.load_font("fonts/fontawesome-webfont.ttf", 24., font_loading_params_large_icon)

    # Load a colored font
    font_loading_params_color = hello_imgui.FontLoadingParams()
    font_loading_params_color.load_color = True
    app_state.color_font = hello_imgui.load_font("fonts/Playbox/Playbox-FREE.otf", 24., font_loading_params_color)

def create_default_docking_splits() -> List[hello_imgui.DockingSplit]:
    #    ____________________________________________
    #    |         |                                |
    #    |         |                                |
    #    | Command |    MainDockSpace (implicit)    |
    #    | Space   |                                |
    #    |         |--------------------------------|
    #    |         |       CommandSpace2            |
    #    --------------------------------------------
    split_main_command = hello_imgui.DockingSplit()
    split_main_command.initial_dock = "MainDockSpace"
    split_main_command.new_dock = "CommandSpace"
    split_main_command.direction = imgui.Dir.left
    split_main_command.ratio = 0.25

    split_main_command2 = hello_imgui.DockingSplit()
    split_main_command2.initial_dock = "MainDockSpace"
    split_main_command2.new_dock = "CommandSpace2"
    split_main_command2.direction = imgui.Dir.right #imgui.Dir.down
    split_main_command2.ratio = 0.5

    return [split_main_command, split_main_command2]

def setup_my_theme():
    hello_imgui.imgui_default_settings.setup_default_imgui_style()
    tweaked_theme = hello_imgui.ImGuiTweakedTheme()
    tweaked_theme.theme = hello_imgui.ImGuiTheme_.material_flat
    tweaked_theme.tweaks.rounding = 10.0
    hello_imgui.apply_tweaked_theme(tweaked_theme)
    imgui.get_style().item_spacing = (6, 4)
    imgui.get_style().set_color_(imgui.Col_.text, ImVec4(0.8, 0.8, 0.85, 1.0))

# Based on:
# https://github.com/pthom/imgui_bundle/blob/main/bindings/pyodide_web_demo/examples/demo_docking.py

if __name__ == "__main__":
    viewer = Viewer()