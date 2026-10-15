"""Build an unregistered FHMODEL2 driver and run scoped, camera-free tests."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'apps/frame-hands'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++', help='C++ compiler executable, or zig/zig.exe')
    parser.add_argument('--predict', action='store_true', help='Compile optional additional PC pose prediction')
    parser.add_argument('--link-type', action='store_true', help='Advertise the experimental Steam Link hand type')
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error('Compiler not found: ' + args.compiler)
    command = [compiler]
    if Path(compiler).stem.lower() == 'zig':
        command.append('c++')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault('ZIG_GLOBAL_CACHE_DIR', str(build / 'zig-global-cache'))
    env.setdefault('ZIG_LOCAL_CACHE_DIR', str(build / 'zig-local-cache'))
    flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-misleading-indentation',
             '-isystem', str(SOURCE / 'system-input/vendor'), '-pthread']
    suffix = '.exe' if os.name == 'nt' else ''
    test_flags = flags + ['-UNDEBUG']  # Zig release modes may otherwise erase assert checks.
    tests = ['test_model_driver', 'test_driver_input_mode', 'test_retarget_continuity', 'test_link_gestures']
    for name in tests:
        exe = build / (name + suffix)
        subprocess.run(command + test_flags + [str(SOURCE / 'pc-input' / (name + '.cpp')), '-o', str(exe)], check=True, env=env)
        subprocess.run([str(exe)], check=True)
    gate = build / ('test_dashboard_gate' + suffix)
    subprocess.run(command + test_flags + [str(SOURCE / 'native/system-driver/controller_watch_v3/test_gate.cpp'), '-o', str(gate)], check=True, env=env)
    subprocess.run([str(gate)], check=True)
    defines = ['MODEL', 'STABILITY', 'ROLE_GRACE', 'INPUT_MODE', 'NAVIGATION', 'DISPLAY', 'GESTURE_CONTROLS']
    if args.predict:
        defines.append('PREDICT')
    if args.link_type:
        defines.append('LINK')
    binary = build / ('driver_fh_skeletal.dll' if os.name == 'nt' else 'driver_fh_skeletal.so')
    shared = ['-shared'] + ([] if os.name == 'nt' else ['-fPIC'])
    subprocess.run(command + flags + shared + ['-DFRAME_HANDS_DRIVER_' + x + '=1' for x in defines]
                   + [str(SOURCE / 'pc-input/driver.cpp'), '-o', str(binary)], check=True, env=env)
    print('Built:', binary)
    print('Passed scoped tests. No driver registered; no hardware accessed.')


if __name__ == '__main__':
    main()
