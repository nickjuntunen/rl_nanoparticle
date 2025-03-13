CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC
INCLUDES = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)
PYTHON_LDFLAGS = $(shell python3-config --ldflags)

# Common source files
COMMON_SRC_FILES = lattice.cpp rate_calculator.cpp variable_field.cpp monte_carlo.cpp utilities.cpp site_info.cpp simulation.cpp
COMMON_SRCS = $(addprefix ./kmc_simulation/, $(COMMON_SRC_FILES)) ./kmc_simulation/run.cpp

# Python module targets
MODULE_SRCS = $(COMMON_SRCS) ./kmc_simulation/bindings.cpp
MODULE_OBJS = $(MODULE_SRCS:.cpp=.o)
EXTENSION_NAME = ./rl/kmc_lattice_gas

# Executable targets
EXE_SRCS = $(COMMON_SRCS)
EXE_OBJS = $(EXE_SRCS:.cpp=.o)
EXE_NAME = ./kmc_simulation/run

# Default target builds both
all: module executable

# Python module
module: $(EXTENSION_NAME)$(shell python3-config --extension-suffix)

$(EXTENSION_NAME)$(shell python3-config --extension-suffix): $(MODULE_OBJS)
	$(CXX) -shared $(CXXFLAGS) -o $@ $(MODULE_OBJS) $(PYTHON_LDFLAGS)

# Standalone executable
executable: $(EXE_NAME)

$(EXE_NAME): $(EXE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(EXE_OBJS) $(PYTHON_LDFLAGS) -lpython3.12

# Common compilation rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f ./kmc_simulation/*.o
	rm -f ./rl/$(EXTENSION_NAME)$(shell python3-config --extension-suffix)
	rm -f ./kmc_simulation$(EXE_NAME)
	rm -f ./kmc_simulation/run