SPDK_DIR ?= /home/wq/spdk
APP_NAME := pca_ann

SRC := \
	main.c \
	pipeline_stage.c \
	query_loader.c

# PKG_CONFIG_PATH := $(SPDK_DIR)/build/lib/pkgconfig
export PKG_CONFIG_PATH=/home/wq/spdk/build/lib/pkgconfig:$PKG_CONFIG_PATH

SPDK_PC_LIBS := spdk_nvme spdk_env_dpdk spdk_event spdk_thread spdk_util

SPDK_CFLAGS := $(shell pkg-config --cflags $(SPDK_PC_LIBS))
SPDK_LIBS   := $(shell pkg-config --libs $(SPDK_PC_LIBS))
SPDK_SYSLIBS := $(shell pkg-config --libs --static spdk_syslibs)

CFLAGS := -std=c11 -O3 -march=native -g -Wall -Wextra -Wpedantic -Wno-unused-parameter \
          -D_GNU_SOURCE\
          -I$(SPDK_DIR)/include $(SPDK_CFLAGS)

BIN_DIR := build/bin


all: $(APP_NAME) ivf_write_disk ivf_write_disk_flex ivf_write_disk_1 ivf_baseline_1 ivf_baseline_2

$(APP_NAME): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ $(SRC) -pthread \
	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) \
	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_write_disk: ivf_write_disk.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ ivf_write_disk.c -pthread \
	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) \
	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_write_disk_flex: ivf_write_disk_flex.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ ivf_write_disk_flex.c -pthread \
	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) \
	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_write_disk_1: ivf_write_disk_1.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ ivf_write_disk_1.c -pthread \
	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) \
	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_baseline_1: ivf_baseline_1.c query_loader.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ ivf_baseline_1.c query_loader.c -lm -pthread 	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) 	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

ivf_baseline_2: ivf_baseline_2.c query_loader.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ ivf_baseline_2.c query_loader.c -lm -pthread \
	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) \
	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

clean:
	rm -f $(BIN_DIR)/$(APP_NAME) $(BIN_DIR)/ivf_write_disk $(BIN_DIR)/ivf_write_disk_flex $(BIN_DIR)/ivf_write_disk_1 $(BIN_DIR)/ivf_baseline_1 $(BIN_DIR)/ivf_baseline_2
 
