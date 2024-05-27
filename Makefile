DIR:=$(shell pwd)
DEVICE?=cpu
RUNTIME?=runc
FLAGS?=DEVICE=$(DEVICE) RUNTIME=$(RUNTIME)


all: help

.PHONY: help
#> Display this message and exit
help:
	@echo "Commands:"
	@awk 'match($$0, "^#>") { sub(/^#>/, "", $$0); doc=$$0; getline; split($$0, c, ":"); cmd=c[1]; print "  \033[00;32m"cmd"\033[0m"":"doc }' ${MAKEFILE_LIST} | column -t -s ":"

.PHONY: install
#> Install dependencies
install:
	poetry install

.PHONY: docker.elliot
#> Build Elliot docker image
docker.elliot:
	docker build -t elliot:latest -f docker/elliot.Dockerfile .

.PHONY: docker.mymedialite
#> Build MyMediaLite docker image
docker.mymedialite:
	docker build -t mymedialite:latest -f docker/mymedialite.Dockerfile .

.PHONY: dco.up
#> Run docker-compose instance
dco.up: dco.kill
	@$(FLAGS) docker-compose up --build --force-recreate --remove-orphans -d exp

.PHONY: dco.kill
#> Kill docker-compose instance
dco.kill:
	@$(FLAGS) docker-compose rm --stop --force exp

.PHONY: clean
#> Clean cached files
clean:
	@find . -type d -name "__pycache__" -not -path '*/.venv/*' -exec rm -rf {} \+
