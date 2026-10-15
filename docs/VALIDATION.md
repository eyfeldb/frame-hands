# Validation

The source package was checked on October 14, 2026 with Windows, Python 3.12.14, and Zig 0.15.2:

- Six exporter tests and seven native-to-PC transport tests passed.
- Python syntax, JSON parsing, file selection, and scans for known credential patterns and personal paths passed.
- Five C++ test programs passed with assertions enabled: model driver, input ownership, retarget continuity, gesture values, and dashboard gate.
- The Windows FHMODEL2 driver built successfully. It wasn't installed or loaded into SteamVR during these checks.
- The controller watcher compiled to an ARM64 Linux object using the included headers. It wasn't run on a headset during packaging.

One older non-model test configuration produces unused-field warnings. Dashboard Python service tests require Linux and were skipped on Windows. The GitHub Actions workflow runs those tests and builds the watcher on Linux; its results are available in the repository's Actions tab once run.

These checks don't measure tracking accuracy or validate a fresh headset installation. See [status](STATUS.md) for the hardware results and remaining work.
