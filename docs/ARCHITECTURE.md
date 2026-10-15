# How it works

```text
Frame cameras -> native tracker -> stabilized world-space joints
                                          |
                         +----------------+----------------+
                         |                                 |
                 native SteamVR driver                pose export
                         |                                 |
                 headset apps / gloves          PC relay + alignment
                                                           |
                                                   PC SteamVR driver
                                                           |
                                                          game
```

FHMODEL2 sends 21 joint positions per hand, capture and publication timestamps, hand identities, and an expiry time. The driver converts those joints into SteamVR's 31-bone skeleton. The additional bones are inferred. The PC relay uses the headset poses from both runtimes to align their tracking spaces.

FHMODE1 selects whether Hands or Controllers owns input. Losing sight of a hand doesn't itself switch to Controllers. The native watcher and PC companion handle controller input, while the dashboard icon reflects the state confirmed by the native service.

Camera access and network transport run outside the SteamVR server driver. Diagnostic recorders, replay scripts, and calibration tools are started separately.

The glove renderer and each game's hand renderer can interpret the same skeleton differently. When a game looks wrong, compare its device pose, bone transforms, and timing with the native output before changing smoothing or tracking.
