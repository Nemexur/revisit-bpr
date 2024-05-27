#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat << EOF
Usage: $(basename "${BASH_SOURCE[0]}") [OPTIONS] DATASET_PATH

Create one csv file for Netflix Prize dataset
from training_set directory.

Options:
  -mr, --min-rating    Minimal rating value
  -h,  --help          Print help
EOF
    exit
}

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1}
    msg "$msg"
    exit "$code"
}

parse_params() {
    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -mr | --min-rating)
            min_rating="${2-}"
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done

    args=("$@")

    [[ ${#args[@]} -eq 0 ]] && die "Missing script arguments"

    return 0
}

min_rating="0"

parse_params "$@"

dir=${args[0]}
ratings="$(find "${dir}" -type f | sort)"
echo "user,rating,date,movie"
paste <(echo "${ratings}") <(seq "$(echo "${ratings}" | wc -l)") | while read -r file movie; do
    tail -n +2 "${file}" | awk -v mr="${min_rating}" 'BEGIN { FS = "," } $2 >= mr' | sed "s/$/,${movie}/"
done
