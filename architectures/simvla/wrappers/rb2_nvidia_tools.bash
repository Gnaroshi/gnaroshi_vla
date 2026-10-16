# User-local diagnostic commands; sourced by rb2 ~/.bash_aliases.
nvidia-smi() {
    /usr/bin/python3 "$HOME/private/gnaroshi_vla_storage/tools/nvidia_user_runtime.py" /usr/bin/nvidia-smi "$@"
}
nvitop() {
    /usr/bin/python3 "$HOME/private/gnaroshi_vla_storage/tools/nvidia_user_runtime.py" /usr/bin/nvitop "$@"
}
