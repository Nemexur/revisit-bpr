#!/usr/bin/env bash

# shellcheck disable=SC2016

set -Eeuo pipefail

script_name=$(basename "${BASH_SOURCE[0]}")
args=(FILE)

usage() {
    cat << EOF
Format jsonl datasets to recbole format.

USAGE: ${script_name} [OPTIONS] ${args[*]}

OPTIONS:
  -g, --grouped    Processe grouped dataset
  -h, --help       Print help.
EOF
    exit
}

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1}
    msg "${script_name}: ${msg}"
    exit "${code}"
}

parse_params() {
    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -g | --grouped)
            grouped=true
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done
    argv=("$@")
    paste <(tr " " "\n" <<< "${args[@]}") <(tr " " "\n" <<< "${argv[@]:-""}") | while read -r arg value; do
        if [ -z "${value}" ]; then
            die "Missing argument: ${arg}"
        fi
    done
    return 0
}

grouped=false

parse_params "$@"

file_path=${argv[0]}

printf "user_id:token\titem_id:token\trating:float\n"
if ${grouped}; then
    mlr --ijsonl --otsv reorder -f user,item,rating <(mlr --jsonl cat "${file_path}" | jq -c '.user as $user | .item | map({user: $user, item: ., rating: 1}) | .[]') | tail -n +2
else
    mlr --ijsonl --otsv put '$rating = 1' + cut -f user,item,rating + reorder -f user,item,rating "${file_path}" | tail -n +2
fi
