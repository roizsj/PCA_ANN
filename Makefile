SPDK_DIR ?= /home/wq/spdk
APP_NAME := pca_ann
PKG_CONFIG ?= pkg-config
FAISS_DIR ?= /home/zhangshujie/opt/faiss

APP_C_SRCS := 	main.c 	pipeline_stage.c 	query_loader.c

APP_CXX_SRCS := 	coarse_search_faiss.cpp

APP_OBJS := $(addprefix build/obj/,$(APP_C_SRCS:.c=.o)) $(addprefix build/obj/,$(APP_CXX_SRCS:.cpp=.o))

SPDK_PKG_CONFIG_PATH := $(SPDK_DIR)/build/lib/pkgconfig$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH))
export PKG_CONFIG_PATH := $(SPDK_PKG_CONFIG_PATH)

SPDK_PC_LIBS := spdk_nvme spdk_env_dpdk spdk_event spdk_thread spdk_util

SPDK_CFLAGS := $(shell PKG_CONFIG_PATH='$(SPDK_PKG_CONFIG_PATH)' $(PKG_CONFIG) --cflags $(SPDK_PC_LIBS))
SPDK_LIBS   := $(shell PKG_CONFIG_PATH='$(SPDK_PKG_CONFIG_PATH)' $(PKG_CONFIG) --libs $(SPDK_PC_LIBS))
SPDK_SYSLIBS := $(shell PKG_CONFIG_PATH='$(SPDK_PKG_CONFIG_PATH)' $(PKG_CONFIG) --libs --static spdk_syslibs)

CFLAGS := -std=c11 -O3 -march=native -g -Wall -Wextra -Wpedantic -Wno-unused-parameter           -D_GNU_SOURCE           -I. -I$(SPDK_DIR)/include $(SPDK_CFLAGS)
CXXFLAGS := -std=c++17 -O3 -march=native -g -Wall -Wextra -Wno-unused-parameter             -D_GNU_SOURCE             -I. -I$(SPDK_DIR)/include -I$(FAISS_DIR)/include $(SPDK_CFLAGS)
FAISS_LDFLAGS := -L$(FAISS_DIR)/lib -Wl,-rpath,$(FAISS_DIR)/lib
FAISS_LIBS := -lfaiss -lstdc++ -lgomp

BIN_DIR := build/bin
OBJ_DIR := build/obj

all: $(APP_NAME) ivf_write_disk ivf_write_disk_flex ivf_write_disk_1 ivf_write_disk_4way_gist ivf_baseline_1 ivf_baseline_2 ivf_baseline_2_gist ivf_baseline_4way_gist

$(APP_NAME): $(APP_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/$@ $(APP_OBJS) -pthread $(FAISS_LDFLAGS) $(FAISS_LIBS) 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

ivf_write_disk: write_disk/ivf_write_disk.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ write_disk/ivf_write_disk.c -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_write_disk_flex: write_disk/ivf_write_disk_flex.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ write_disk/ivf_write_disk_flex.c -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_write_disk_1: write_disk/ivf_write_disk_1.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ write_disk/ivf_write_disk_1.c -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_write_disk_4way_gist: write_disk/ivf_write_disk_4way_gist.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ write_disk/ivf_write_disk_4way_gist.c -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_baseline_1: baseline/ivf_baseline_1.c query_loader.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ baseline/ivf_baseline_1.c query_loader.c -lm -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_baseline_2: baseline/ivf_baseline_2.c query_loader.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ baseline/ivf_baseline_2.c query_loader.c -lm -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_baseline_2_gist: baseline/ivf_baseline_2_gist.c query_loader.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ baseline/ivf_baseline_2_gist.c query_loader.c -lm -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_baseline_4way_gist: baseline/ivf_baseline_4way_gist.c query_loader.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ baseline/ivf_baseline_4way_gist.c query_loader.c -lm -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

clean:
	rm -f $(BIN_DIR)/$(APP_NAME) $(BIN_DIR)/ivf_write_disk $(BIN_DIR)/ivf_write_disk_flex $(BIN_DIR)/ivf_write_disk_1 $(BIN_DIR)/ivf_write_disk_4way_gist $(BIN_DIR)/ivf_baseline_1 $(BIN_DIR)/ivf_baseline_2 $(BIN_DIR)/ivf_baseline_2_gist $(BIN_DIR)/ivf_baseline_4way_gist
	rm -rf $(OBJ_DIR)
