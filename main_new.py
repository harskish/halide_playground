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
    return subprocess.run([vcvarsall_path, '&&', *args], cwd=cwd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@strict_dataclass
class CommonState(ParamContainer):
    kernel: Param = EnumParam('Kernel', 'fuji_debayer', ['write', 'blur', 'fuji_debayer'])
    out_WH: Param = Int2Param('Output (W, H)', (6032, 4028), 32, 8000)
    input: Param = EnumSliderParam('Input', 'Ueno', ['Ueno', 'Black'])
    

def setup_state():
    self = SimpleNamespace()
    self.state = CommonState()
    self.state_last = None
    self.cache = {}
    self.input_img = np.array(Image.open('examples/ueno.jpg').convert('RGB'))
    self.out_ch = self.input_img.shape[-1]
    
    # UI widgets created dynamically based on HL pipeline metadata
    self.params = {} #: dict[str, Param] = {} # label -> Param
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
    #arr = self.recompile_and_run()
    #if arr is not None:
    #    self.update_image(arr)
    return self


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
        arr = self.recompile_and_run()
        if arr is not None:
            self.update_image(arr)
    
    # Override to be extra wide
    @property
    def toolbar_width(self):
        return int(round(600 * self.v.ui_scale))
    
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
            os.environ['STEM'] = stem
            os.environ['HALIDE_PATH'] = str(hl_root.resolve()) # Windows path syntax

            lib_mtime = libname_abs.stat().st_mtime if libname_abs.is_file() else 0 # modification
            os.makedirs('examples/build/tmp', exist_ok=True)
            
            if platform.system() == 'Windows':
                if len(os.environ['PATH']) > 7_000:
                    print('Warning: PATH very long, nmake might fail')
                ret = run_in_vs2022_cmd('nmake', '/f', 'NMakefile', str(libname_rel), cwd='examples')
                if ret.returncode != 0:
                    err_msg = ret.stderr.decode().strip()
                    #print(f'STDERR:\n"{err_msg}"')
                    return err_msg
                print('STDOUT:', ret.stdout.decode())
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
            return self.recompile_and_run()
        return None # reuse cached

# Links
# https://github.com/bobobo1618/halideraw
# https://github.com/zshipko/halide-runtime
# https://github.com/dragly/numlide
# https://github.com/dragly/halide-widgets (https://github.com/dragly?tab=repositories)
# https://github.com/anuejn/halide_experiments/blob/main/halide_blocks/debayer.py

import json
from enum import Enum
import time

from imgui_bundle import hello_imgui, imgui, ImVec4, ImVec2
from imgui_bundle.demos_python import demo_utils
from typing import List


##########################################################################
#    Our Application State
##########################################################################
from dataclasses import field
@strict_dataclass
class MyAppSettings:
    value: int = 10

class RocketState(Enum):
    Init = 0
    Preparing = 1
    Launched = 2

# Struct that holds the application's state
@strict_dataclass
class AppState:
    f: float = 0
    counter: int = 0
    my_app_settings: MyAppSettings = field(default_factory=lambda : MyAppSettings())
    title_font: imgui.ImFont = None
    color_font: imgui.ImFont = None
    emoji_font: imgui.ImFont = None
    large_icon_font: imgui.ImFont = None

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


# For saving into .ini
def my_app_settings_to_string(settings: MyAppSettings) -> str:
    as_dict = {}
    as_dict["value"] = settings.value
    return json.dumps(as_dict)

def string_to_my_app_settings(s: str) -> MyAppSettings:
    r = MyAppSettings()
    try:
        as_dict = json.loads(s)
        r.value = as_dict["value"]
    except Exception as e:
        hello_imgui.log(hello_imgui.LogLevel.error, f"Error while loading user settings: {e}")
    return r

def load_my_app_settings(app_state: AppState):
    app_state.my_app_settings = string_to_my_app_settings(
        hello_imgui.load_user_pref("MyAppSettings")
    )

def save_my_app_settings(app_state: AppState):
    hello_imgui.save_user_pref(
        "MyAppSettings", my_app_settings_to_string(app_state.my_app_settings)
    )

def draw_output(app_state: AppState):
    imgui.push_font(app_state.title_font)
    imgui.text("Pipeline output here")
    imgui.pop_font()

def draw_inputs(app_state: AppState):
    # emulate C/C++ static variable: we will store some static variables
    # as attributes of the function
    statics = draw_inputs

    # Apply the theme before opening the window
    #tweaked_theme = hello_imgui.ImGuiTweakedTheme()
    #tweaked_theme.theme = hello_imgui.ImGuiTheme_.white_is_white
    #tweaked_theme.tweaks.rounding = 0.0
    #hello_imgui.push_tweaked_theme(tweaked_theme)

    # Open the window
    if imgui.begin("Pipeline inputs"):
        imgui.push_font(app_state.title_font)
        imgui.text("Pipeline inputs")
        imgui.pop_font()
        imgui.text("This window uses a different theme")

    # Close the window
    imgui.end()

    # Restore the theme
    #hello_imgui.pop_tweaked_theme()


def draw_text_editor(app_state: AppState):
    imgui.text('<Kernel name>.cpp')
    imgui.separator()

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

#
def create_dockable_windows(app_state: AppState) -> List[hello_imgui.DockableWindow]:
    # Text editor: left-hand side
    text_editor_window = hello_imgui.DockableWindow()
    text_editor_window.label = "Text editor"
    text_editor_window.dock_space_name = "CommandSpace"
    text_editor_window.gui_function = lambda: draw_text_editor(app_state)

    # Output image: right top
    output_window = hello_imgui.DockableWindow()
    output_window.label = "Pipeline Output"
    output_window.dock_space_name = "MainDockSpace"
    output_window.gui_function = lambda: draw_output(app_state)

    # Param sliders: right bottom
    widgets_window = hello_imgui.DockableWindow()
    widgets_window.call_begin_end = False # calling imgui.{begin/end} manually
    widgets_window.label = "Input Params"
    widgets_window.dock_space_name = "CommandSpace2"
    widgets_window.gui_function = lambda: draw_inputs(app_state)

    dockable_windows = [
        text_editor_window,
        output_window,
        widgets_window,
    ]
    return dockable_windows

def create_default_layout(app_state: AppState) -> hello_imgui.DockingParams:
    docking_params = hello_imgui.DockingParams()
    docking_params.docking_splits = create_default_docking_splits()
    docking_params.dockable_windows = create_dockable_windows(app_state)
    docking_params.main_dock_space_node_flags |= imgui.DockNodeFlags_.auto_hide_tab_bar
    return docking_params

def setup_my_theme():
    hello_imgui.imgui_default_settings.setup_default_imgui_style()
    tweaked_theme = hello_imgui.ImGuiTweakedTheme()
    tweaked_theme.theme = hello_imgui.ImGuiTheme_.material_flat
    tweaked_theme.tweaks.rounding = 10.0
    hello_imgui.apply_tweaked_theme(tweaked_theme)
    imgui.get_style().item_spacing = (6, 4)
    imgui.get_style().set_color_(imgui.Col_.text, ImVec4(0.8, 0.8, 0.85, 1.0))

def main():
    # Installed by pip, includes two fonts
    hello_imgui.set_assets_folder(demo_utils.demos_assets_folder())

    # Our application state
    app_state = AppState()
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Halide Playground"
    runner_params.imgui_window_params.menu_app_title = "HLP"
    runner_params.app_window_params.window_geometry.size = (1000, 900)
    runner_params.app_window_params.restore_previous_geometry = True
    runner_params.callbacks.post_init = lambda: load_my_app_settings(app_state)
    runner_params.callbacks.before_exit = lambda: save_my_app_settings(app_state)

    # HDR output (only available on Metal)
    # TODO: probably need to set metal backend first
    if hello_imgui.has_edr_support():
        runner_params.renderer_backend_type = hello_imgui.RendererBackendType.metal
        renderer_backend_options = hello_imgui.RendererBackendOptions()
        renderer_backend_options.request_float_buffer = True
    
    # Fonts
    runner_params.callbacks.load_additional_fonts = lambda: load_fonts(app_state)
    
    # Status bar: fps etc.
    runner_params.imgui_window_params.show_status_bar = False
    #runner_params.callbacks.show_status = lambda: status_bar_gui(app_state)

    # Change style
    runner_params.callbacks.setup_imgui_style = setup_my_theme

    # Create "MainDockSpace"
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )

    # Allow splitting into separate windows?
    runner_params.imgui_window_params.enable_viewports = True
    
    # Default layout
    runner_params.docking_params = create_default_layout(app_state)
    runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    runner_params.ini_filename = 'halide_playground.ini'
    ini_path = os.path.join(hello_imgui.ini_folder_location(runner_params.ini_folder_type), runner_params.ini_filename)
    print(f'INI path: {ini_path}')

    hello_imgui.run(runner_params)

# Based on:
# https://github.com/pthom/imgui_bundle/blob/main/bindings/pyodide_web_demo/examples/demo_docking.py

if __name__ == "__main__":
    main()