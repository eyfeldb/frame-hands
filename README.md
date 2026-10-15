# Frame Hands

Hand tracking for Steam Frame, with a SteamVR driver, a Windows PC bridge, and a Hands toggle in the headset dashboard. Tracking runs on the headset; the bridge sends hand poses to SteamVR on your PC.

This is experimental developer software. It has been used on one Frame, but setup on a fresh headset still needs work. The PC driver and tests build from this repository; the full tracker also needs models and runtime assets that are not included. Start with the [build instructions](docs/BUILDING.md) and [known limits](docs/STATUS.md).

## What's included

All project code is under `apps/frame-hands/`:

- `native/`: tracking, pose processing, native service, and pose transport.
- `pc-input/`: SteamVR driver, hand retargeting, bindings, gesture controls, and tests.
- `hand-display/`: headset and Windows glove renderers.
- `dashboard/`: Hands icon and service integration.
- `pc-input/maestro-audit/`: headset video recorder.
- `exploratory/`, `training/`, `viewer/`, `camera-recovery/`, and `system-input/`: experiments and development tools. Some still depend on the original development setup.

See [architecture](docs/ARCHITECTURE.md) for how the pieces fit together.

## Build and test

You'll need Python 3.10+ and a C++17 compiler. Run these from the repository root:

```sh
python tools/check_source.py
python tools/build_driver.py --compiler c++
```

On Windows, you can use Zig 0.15.2:

```powershell
python tools/build_driver.py --compiler C:/tools/zig/zig.exe
```

These commands check the source, run tests, and build a driver under `build/`. They don't install it or start tracking. No headset is needed. Dashboard service tests run on Linux and are skipped on Windows. See [validation](docs/VALIDATION.md) for the results so far.

## Data and license

Recordings, personal hand profiles, trained weights, device settings, and installed binaries are excluded. The [build guide](docs/BUILDING.md) lists the missing dependencies; [privacy notes](PRIVACY.md) explain the recording tools.

Original code is MIT licensed. See [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies and asset provenance. `SOURCE-MANIFEST.json` records the source revision and file hashes. This repository starts with a fresh Git history.

Frame Hands is an independent project and is not affiliated with Valve.
