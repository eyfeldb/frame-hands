# Building Frame Hands

## PC driver and tests

From the repository root, with Python 3.10+ and a C++17 compiler:

```sh
python tools/check_source.py
python tools/build_driver.py --compiler c++
```

On Windows, pass the path to a compatible compiler or Zig 0.15.2's `zig.exe`. The helper adds `c++` when using Zig. Output goes into `build/`, which Git ignores. OpenVR headers and their license are included.

The default build uses FHMODEL2 and includes input ownership, hand-role retention, navigation, display support, and gesture controls. `--predict` enables extra pose prediction; `--link-type` changes the Steam Link device identity. Both are experimental options.

The helper runs tests and builds the driver without registering it in SteamVR.

## Headset tracker

**The full tracker cannot yet be installed from this repository alone.** Several older preparation and deployment scripts depend on local checkpoints and manifests from the development machine. Those files aren't included, so those scripts aren't a fresh-install guide.

A complete setup still needs:

1. A pinned Frametop checkout, camera reader patches, and an ncnn build.
2. Compatible palm and hand model weights, including the history-conditioned model.
3. Generic rig profiles and Godot pose-processing assets to replace personal profiles and trial snapshots.
4. Frame's ARM64 runtime libraries and the camera reader's required permissions, verified for the installed runtime.
5. SteamVR glove assets for the optional hand display. `hand-display/export_assets.py` extracts these from a local SteamVR installation. Asset redistribution needs a separate review before distributing binaries.

The tracker targets Linux ARM64 on Frame. The Windows driver consumes its output. Training and replay tools need additional scientific and ML packages that the basic build doesn't install.

## Dashboard toggle

`apps/frame-hands/dashboard/install.py` expects the native Hands service to be installed already. The adapter checks Steam UI build `11094443` and its asset hashes, and refuses an unfamiliar build. Its device path is `~/devkit-game/FrameHandsDashboard-v1`.

The controller watcher is in `native/system-driver/controller_watch_v3/`, with its shared headers included. To compile it on Linux:

```sh
sh apps/frame-hands/native/system-driver/controller_watch_v3/build_source.sh
```

This command builds the watcher without changing an installed package. The older `build_headset.sh` expects an existing prepared package. Dashboard installation on a fresh headset still needs testing.

## Headset recorder

`pc-input/maestro-audit/record_headset.cpp` and `gpu_readback.h` require OpenVR, Vulkan headers and libraries, and FFmpeg. Audio cues use `paplay`, with `start.wav` and `done.wav` supplied in the parent capture folder.

The recorder captures the composited headset view. Camera images and synchronized tracker/PC poses require separate capture tools. Review the recorded video and timestamps to check what was actually captured.
