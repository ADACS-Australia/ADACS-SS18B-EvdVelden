# This makefile combines the targets of the various
# codes implemented in this project into a unified set

# Set the default target to 'build'
.PHONY: default
default: build

# This ensures that we use standard (what is used in interactive shells) version of echo.
ECHO = /bin/echo
ECHO_NNL = /bin/echo -n
export ECHO
export ECHO_N

# Extract the project name from the parent directory
PRJ_DIR=$(PWD)
PRJ_NAME=`grep "name" .project.json | awk '{print $$2}' | sed -e 's/^"//' -e 's/"$$//'`

# Get git hash
ifeq ($(shell git log --oneline -5 2>/dev/null | wc -l ),0)
	GIT_HASH="No commits"
else
	GIT_HASH=$(shell git rev-parse --short HEAD)
endif

# Fetch the version from the .version file
ifneq ($(wildcard .version),)
	PRJ_VERSION:=`cat .version`
	PRJ_VERSION:='v'$(PRJ_VERSION)
else
	PRJ_VERSION:=unset
endif

# Set targets to check for needed executables
list_of_checks = pigar
.PHONY: $(list_of_checks)
$(list_of_checks): % : check-%-in-path
check-%-in-path:
	@$(if $(shell which $* >& /dev/null ; if [ $$? -ne 0 ] ; then echo fail ; fi),$(error '$*' not in path.  Please (re)install and try command again if you want this Makefile target's support))

# Determine what sort of environment we're in (eg. OSX or Linux)
OSTYPE := $(word 1,$(shell uname -msr))
_MAC_BUILD=0
_TRAVIS_BUILD=0
ifdef TRAVIS_OS_NAME
    ifeq ($(TRAVIS_OS_NAME),'osx')
        _MAC_BUILD=1
    endif
    _TRAVIS_BUILD=1
else
    ifeq (${OSTYPE},Darwin)
        _MAC_BUILD=1
    endif
endif
export _MAC_BUILD
export _TRAVIS_BUILD

# A directory for storing stuff related to testing
TESTS_DIR = '.tests'

# Coverage paths
KCOV_DIR=$(TESTS_DIR)'/kcov-master'
KCOV_EXE=${PWD}'/'$(KCOV_DIR)'/build/src/Release/kcov'
export KCOV_EXE
export KCOV_DIR

# The build directory for documentation ('_build' to avoid breaking Readthedocs builds)
BUILD_DIR_DOCS:=$(PRJ_DIR)/docs/_build

# List of common targets (potentially) requiring specialized action for each language separately
BUILD_LIST =
DOCS_LIST = 
INSTALL_LIST = 
TEST_LIST = 
COVERAGE_LIST = 
CLEAN_LIST = 
LINT_LIST = 

# Add appropriate targets if C/C++ is supported by this project
ifneq ($(wildcard .Makefile-c),)
	include .Makefile-c
	BUILD_LIST := $(BUILD_LIST) build-c
	DOCS_LIST := $(DOCS_LIST) build docs-c
	INSTALL_LIST := $(INSTALL_LIST) install-c
	TEST_LIST := $(TEST_LIST) tests-c
	COVERAGE_LIST := $(COVERAGE_LIST) coverage-c
	CLEAN_LIST := $(CLEAN_LIST) clean-c
	LINT_CHECK_LIST := $(LINT_CHECK_LIST) lint-check-c
	LINT_FIX_LIST := $(LINT_FIX_LIST) lint-fix-c
endif

# Add appropriate targets if Python is supported by this project
ifneq ($(wildcard .Makefile-py),)
	include .Makefile-py
	BUILD_LIST := $(BUILD_LIST) build-py
	INSTALL_LIST := $(INSTALL_LIST) install-py
	TEST_LIST := $(TEST_LIST) tests-py
	COVERAGE_LIST := $(COVERAGE_LIST) coverage-py
	CLEAN_LIST := $(CLEAN_LIST) clean-py
	LINT_CHECK_LIST := $(LINT_CHECK_LIST) lint-check-py
	LINT_FIX_LIST := $(LINT_FIX_LIST) lint-fix-py
endif

#############################
# Targets for project users #
#############################

# Help
help:
	@$(ECHO) "Usage: make [TARGETS]"
	@$(ECHO) 
	@$(ECHO) "Available targets:"
	@$(ECHO) "	init      - perform project initialization [run once before anything else]"
	@$(ECHO) "	build     - build all project software"
	@$(ECHO) "	install   - install all project software"
	@$(ECHO) "	docs      - build documentation"
	@$(ECHO) "	docs-open - open documentation in the browser"
	@$(ECHO) "	docs      - build documentation"
	@$(ECHO) "	clean     - clean-out unneeded build/development/etc files"
	@$(ECHO) 
	@$(ECHO) "Additional targets for development:"
	@$(ECHO) "	tests               - run all project tests"
	@$(ECHO) "	tests-py-tox        - run all python tests under tox"
#	@$(ECHO) "	coverage            - build code coverage reports for tests"
	@$(ECHO) "	lint-check          - check the project linting standards"
	@$(ECHO) "	lint-fix            - apply the project linting standards"
	@$(ECHO) "	docs-update         - update API documentation"
	@$(ECHO) "	requirements-update - rebuild the project's '.requirements.txt' file"
	@$(ECHO) 

# One-time initialization
.PHONY: init
init:	 .print_status submodules requirements

# Build project
.PHONY: build
build:	 .print_status $(BUILD_LIST)

# Install project
.PHONY: install
install: .print_status $(INSTALL_LIST)

# Full-build
.PHONY: all
all:	.print_status init build install

# Clean project
.PHONY: clean
clean:	 .print_status $(CLEAN_LIST) project-clean

# Make sure all submodules are installed
.PHONY: submodules
submodules:
	@$(ECHO_NNL) "Checking that all git submodules are up-to-date..."
	@git submodule update --recursive
	@$(ECHO) "Done."

# Make sure all needed Python code has been installed into the current environment
.PHONY: requirements
requirements:
	@$(ECHO_NNL) "Making sure that all needed Python modules are present..."
ifeq ($(shell which python),)
	@$(error "'python' not in path.  Please install it or fix your environment and try again.)
endif
ifeq ($(shell which pip),)
	@$(error "'pip' not in path.  Please install it or fix your environment and try again.)
endif
	@pip -q install --src .src -r .requirements.txt
	@pip -q install --src .src -r .requirements_dev.txt
	@$(ECHO) "Done."

########################################
# Targets for generating documentation #
########################################

# Build the project documentation
.PHONY: docs
docs: $(DOCS_LIST) docs-update
	@$(ECHO_NNL) "Building documentation..."
	@cd docs;sphinx-build . _build
	@$(ECHO) "Done."

# Update API documentation
# n.b.: 'build' is a dependency because a build needs to be in place
#       in order to generate executable syntax documentation.
.PHONY: docs-update
docs-update: build $(BUILD_DIR_DOCS)
	@$(ECHO_NNL) "Updating API documentation..."
	@$(PRJ_NAME)_helper update_docs
	@$(ECHO) "Done."

# Open the documentation
.PHONY: docs-update
docs-open:
	@$(ECHO_NNL) "Opening documentation in browser..."
	@open ${BUILD_DIR_DOCS}/index.html
	@$(ECHO) "Done."

# Make the documentation build directory
$(BUILD_DIR_DOCS):
ifeq (,$(wildcard $@))
	@$(ECHO_NNL) -n "Making docs build directory {"$@"}..."
	@mkdir $@
	@$(ECHO) "Done."
endif

# Perform all build-system cleaning
.PHONY: project-clean project-clean-start project-clean-stop
project-clean-start:
	@$(ECHO_NNL) "Cleaning-up project debris..."
project-clean-stop:
	@$(ECHO) "Done."
project-clean: project-clean-start docs-clean tests-clean project-clean-stop

# Remove the documentation build directory
.PHONY: docs-clean
docs-clean:
	@rm -rf docs/__pycache__
	@rm -rf $(BUILD_DIR_DOCS)

# Remove the tests directory
.PHONY: docs-clean
tests-clean:
	@rm -rf $(TESTS_DIR)

##################################
# Targets for project developers #
##################################

# Run tests
tests:	.print_status $(TEST_LIST)

# Generate code coverage reports
coverage:	.print_status build $(COVERAGE_LIST)

# Make linting suggestions
lint-check:	.print_status $(LINT_CHECK_LIST)

# Apply all linting suggestions
lint-fix:	.print_status $(LINT_FIX_LIST)

# Update the pip python requirements files for the project.
.PHONY: requirements-update
requirements-update: .print_status check-pigar-in-path
	@$(ECHO_NNL) "Updating project Python requirements..."
	@pigar -p .requirements.txt
	@$(ECHO) "Done."

# Download kcov
.PHONY: $(TESTS_DIR)/kcov.tgz
$(TESTS_DIR)/kcov.tgz:
	@$(ECHO_NNL) "Downloading kcov code..."
ifneq ($(wildcard $(TESTS_DIR)/.),)	
	@mkdir $(TESTS_DIR)
endif
	@wget https://github.com/SimonKagstrom/kcov/archive/master.tar.gz -O $(TESTS_DIR)/kcov.tgz
	@tar xfz $(TESTS_DIR)/kcov.tgz -C $(TESTS_DIR)
	@$(ECHO) "Done."

# Build kcov
.PHONY: $(KCOV_EXE)
$(KCOV_EXE): $(TESTS_DIR)/kcov.tgz
ifeq ($(_MAC_BUILD),1)
	@cd $(KCOV_DIR);mkdir build;cd build;cmake -G Xcode .. ;xcodebuild -configuration Release
else
	@cd $(KCOV_DIR);mkdir build;cd build;cmake .. ;make
endif

# Generate and upload coverage information
.PHONY: kcov
kcov: $(KCOV_EXE) $(COVERAGE_LIST)
	@$(ECHO_NNL) "Finalizing Codecov integration..."
	@bash -c "bash <(curl -s https://codecov.io/bash) -t $(TOKEN_KCOV)"
	@$(ECHO) "Done."

##########################
# Print a status message #
##########################
.print_status: .printed_status
# Fetch the version from the .version file
	@$(ECHO)
	@$(ECHO) "Project information:"
	@$(ECHO) "-------------------"
	@$(ECHO) "Project name:     "$(PRJ_NAME)
	@$(ECHO) "Project version:  "$(PRJ_VERSION)
	@$(ECHO) "Git hash (short): "$(GIT_HASH)
ifeq ($(_MAC_BUILD),1)
ifeq ($(_TRAVIS_BUILD),1)
	@$(ECHO) "Detected system:  Mac on Travis"
else
	@$(ECHO) "Detected system:  Mac"
endif
else
ifeq ($(_TRAVIS_BUILD),1)
	@$(ECHO) "Detected system:  Travis"
else
	@$(ECHO) "Detected system:  Default (Linux assumed)"
endif
endif
	@$(ECHO)
	@rm -rf .printed_status
.printed_status:
	@touch .printed_status
