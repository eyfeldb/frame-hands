# Optional PC hand provider. The app binary and scene remain native on Frame.
if [ -f "$HOME/.local/share/frame-hands/pc-piano/rig.json" ]; then
    export FRAME_UME_RIG="$HOME/.local/share/frame-hands/pc-piano/rig.json"
    export FRAME_UME_VIEW="$XDG_RUNTIME_DIR/frametop-hands/pc-piano/pose.json"
fi
