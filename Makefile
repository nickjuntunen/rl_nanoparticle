CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC
INCLUDES = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)

# Common source files
COMMON_SRCS = lattice.cpp variable_field.cpp rate_calculator.cpp monte_carlo.cpp utilities.cpp site_info.cpp simulation.cpp

# Python module targets
MODULE_SRCS = $(COMMON_SRCS) bindings.cpp
MODULE_OBJS = $(MODULE_SRCS:.cpp=.o)
EXTENSION_NAME = kmc_lattice_gas

# Executable targets
EXE_SRCS = $(COMMON_SRCS) run.cpp
EXE_OBJS = $(EXE_SRCS:.cpp=.o)
EXE_NAME = run

# Default target builds both
all: module executable

# Python module
module: $(EXTENSION_NAME)$(shell python3-config --extension-suffix)

$(EXTENSION_NAME)$(shell python3-config --extension-suffix): $(MODULE_OBJS)
	$(CXX) -shared $(CXXFLAGS) -o $@ $(MODULE_OBJS)

# Standalone executable
executable: $(EXE_NAME)

$(EXE_NAME): $(EXE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(EXE_OBJS)

# Common compilation rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(EXTENSION_NAME)$(shell python3-config --extension-suffix)
	rm -f $(EXE_NAME)