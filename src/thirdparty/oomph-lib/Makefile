
SRC_DIR = ./include/
OBJ_DIR = ./lib/

SRCS=$(wildcard $(SRC_DIR)/*.cc)
SRCS := $(filter-out $(SRC_DIR)/tree.template.cc, $(SRCS))
SRCS := $(filter-out $(SRC_DIR)/multi_domain.template.cc, $(SRCS))
SRCS := $(filter-out $(SRC_DIR)/refineable_mesh.template.cc, $(SRCS))

CSRCS=$(wildcard $(SRC_DIR)/*.c)


OBJS=$(SRCS:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

CSRCS=$(wildcard $(SRC_DIR)/*.c)
OBJS:=$(OBJS) $(CSRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
SRCS:=$(SRCS) $(CSRCS)

DEPS=$(OBJS:.o=.d)

LIB_GENERIC= $(OBJ_DIR)/libgeneric.a




PYOOMPH_CONFIG_FILE ?= ../../../pyoomph_config.env
include $(PYOOMPH_CONFIG_FILE)



ifneq ($(PYOOMPH_MARCH_NATIVE),false)
#-Wdate-time
CXXFLAGS =   -fwrapv -O2 -Wall -Wformat -Werror=format-security  -D_FORTIFY_SOURCE=2 -fPIC -DVERSION_INFO="0.0.1" -DNDEBUG 
ifeq ($(OS),Windows_NT)
 CXXFLAGS+=-fno-stack-protector
else
ifeq ($(shell uname),Darwin)
 CXXFLAGS+= -fstack-protector-strong    -march=native
else
 CXXFLAGS+= -fstack-protector-strong   -fvisibility=hidden -march=native
endif
endif
else
#-Wdate-time
CXXFLAGS = -fwrapv -O2 -Wall -Wformat -Werror=format-security  -D_FORTIFY_SOURCE=2 -fPIC -DVERSION_INFO="0.0.1" -DNDEBUG 
ifeq ($(OS),Windows_NT)
 CXXFLAGS+=-fno-stack-protector
else
ifeq ($(shell uname),Darwin)
 CXXFLAGS+=-fstack-protector-strong   
else
 CXXFLAGS+=-fstack-protector-strong -fvisibility=hidden
endif
endif
endif

ifeq ($(PYOOMPH_PARANOID),true)
CXXFLAGS+=-DPARANOID
endif

# Include the MPI wrapping
CXXFLAGS+=  -I../.. -DOOMPH_HAS_MPI

ifneq ($(PYOOMPH_DEBUG_INFOS),false)
 CXXFLAGS+=-g
else
 CXXFLAGS+=-g0
endif

CXXFLAGS+=-std=c++11





.PHONY: all clean
all: $(LIB_GENERIC)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@$(CXX) -MM -MT $@ $(CXXFLAGS) $< > $(OBJ_DIR)/$*.d

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@$(CXX) -MM -MT $@ $(CXXFLAGS) $< > $(OBJ_DIR)/$*.d


$(LIB_GENERIC): $(OBJS)
	echo ar rcs $@ $^
	ar rvs $@ $^

clean:
	rm -rf $(OBJ_DIR)/*.d $(OBJ_DIR)/*.o
