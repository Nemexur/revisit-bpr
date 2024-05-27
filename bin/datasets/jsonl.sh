#!/usr/bin/env bash

set -Eeuo pipefail

script_name=$(basename "${BASH_SOURCE[0]}")
args=(DATASET_PATH)

usage() {
    cat << EOF
Usage: ${script_name} [OPTIONS] ${args[*]}

Convert CSV dataset to jsonlines.

Options:
  -g, --group       Group by user column
  -u, --user-col    User column. Default is user
  -i, --item-col    Item column. Default is item
  -h, --help        Print help
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
        -g | --group)
            group=true
            ;;
        -u | --user-col)
            user_col="${2-}"
            shift
            ;;
        -i | --item-col)
            item_col="${2-}"
            shift
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

if ! command -v jq &> /dev/null; then
    die "jq is not installed. Aborting"
fi
if ! command -v mlr &> /dev/null; then
    die "miller (mlr) is not installed. Aborting"
fi

group=false
user_col="user"
item_col="item"

parse_params "$@"

dataset_path="${argv[0]}"
cmd="mlr --icsv --ojson cat ${dataset_path}"
if ${group}; then
    cmd="${cmd} | jq -c 'group_by(.${user_col}) | map({${user_col}: (.[0].${user_col}), ${item_col}: [.[] | .${item_col}] })[]'"
else
    cmd="${cmd} | jq -c '.[]'"
fi
bash <<< "${cmd}"
