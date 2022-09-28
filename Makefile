###########################
###### EVOS Makefile ######
###########################

ifeq "" "$(EVOS_PYTHONBIN)"
	EVOS_PYTHONBIN=python3
endif

# Clean package
#############################################################################
.PHONY: clean 

clean: 
	@rm -r build 1>/dev/null 2>/dev/null || true
	@rm -r dist 1>/dev/null 2>/dev/null || true 

# Install package
#############################################################################
.PHONY: build 

build:
	@echo "------------------------------------------------ building package ------------------------------------------------"
	@echo ""
	${EVOS_PYTHONBIN} -m pip install .


# create documentation
#############################################################################
.PHONY: doc 

doc:
	pdoc evos &

# perform unit tests
#############################################################################
.PHONY: pytest

pytest:
	${EVOS_PYTHONBIN} -m pytest -rfs -v --strict-markers --showlocals --full-trace || true 

all: clean build

install : all

test: pytest
