# Status and known limits

Updated October 15, 2026. The exact source revision is in `SOURCE-MANIFEST.json`.

Native tracking, SteamVR hand input, and the dashboard toggle have been tested on one Frame. The dashboard uses Steam's UI components through an experimental adapter tied to a specific UI build.

The October 12 tests used SteamOS 0.4.5, build `20261007.6125817`, and native SteamVR 2.18.2, build marker `1791331621`. The PC-streamed Maestro investigation used game build `24847949`.

## Maestro

In the October 12 tests, Maestro's hands followed less accurately than the native glove display. Early PC processing changes reduced measured differences but didn't resolve the visible problem. A 20-second headset video was captured, but it lacked synchronized camera images and pose traces needed to isolate the cause.

This source includes later device-pose corrections, an optional Steam Link hand identity, and revised dashboard/controller handling. The final result in Maestro still needs verification.

## Still to do

- Make model and runtime setup reproducible on a fresh headset.
- Test installation on another device and compatibility with more games.
- Review model and derived-asset redistribution before shipping binaries.
- Measure physical accuracy, motion-to-photon latency, and sustained thermal load.

The automated tests cover protocols and driver behavior with synthetic inputs. Live tracking and compatibility after runtime updates need hardware testing.
