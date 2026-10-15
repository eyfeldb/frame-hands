# Third-party notices

Original Frame Hands code is available under the root [MIT license](LICENSE). Third-party code keeps its own license:

- **Frametop, DeeJanuz (2026):** MIT. The notice is in `apps/frame-hands/system-input/vendor/LICENSE-Frametop` and applies to copied upstream code and modifications. The full upstream source and runtime are not included.
- **OpenVR, Valve:** SDK headers are included with the notice in `apps/frame-hands/system-input/vendor/LICENSE-OpenVR`. Keep that notice when redistributing them.
- **Mercury/Monado helpers, Collabora (2021–2023):** `exploratory/mercury_keypoints.py` and `exploratory/mercury_probe.py` retain their BSL-1.0 identifiers and copyright notices. The full license is in `licenses/BSL-1.0.txt`, copied from the [official Boost license](https://www.boost.org/LICENSE_1_0.txt) on October 14, 2026.

SteamVR, Godot, ncnn, Vulkan, FFmpeg, and optional ML tools are separate dependencies. Their licenses apply when assembling a binary package.

SteamVR glove meshes, model weights, training data, and personal profiles are excluded. `apps/frame-hands/pc-input/reference.h` contains numeric skeleton rest transforms derived from local SteamVR assets. Its asset provenance is separate from the OpenVR SDK license; redistribution terms still need review before a binary release.

Frame Hands is not affiliated with or endorsed by Valve.
