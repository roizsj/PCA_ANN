SPDK_DIR ?= /home/wq/spdk
APP_NAME := pca_ann

SRC := \
	main.c \
	layout.c \
	distance.c \
	query_ctx.c \
	topk.c \

PKG_CONFIG_PATH := $(SPDK_DIR)/build/lib/pkgconfig
export PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/home/wq/spdk/build/lib/pkgconfig:$PKG_CONFIG_PATH

SPDK_PC_LIBS := spdk_nvme spdk_env_dpdk spdk_event spdk_thread spdk_util

SPDK_CFLAGS := $(shell pkg-config --cflags $(SPDK_PC_LIBS))
SPDK_LIBS   := $(shell pkg-config --libs $(SPDK_PC_LIBS))
SPDK_SYSLIBS := $(shell pkg-config --libs --static spdk_syslibs)

CFLAGS := -std=c11 -O0 -g -Wall -Wextra -Wpedantic -Wno-unused-parameter \
          -D_GNU_SOURCE\
          -I$(SPDK_DIR)/include $(SPDK_CFLAGS)

$(APP_NAME): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) -pthread \
	-Wl,--whole-archive -Wl,-Bstatic $(SPDK_LIBS) \
	-Wl,-Bdynamic -Wl,--no-whole-archive $(SPDK_SYSLIBS)

all: $(APP_NAME)

clean:
	rm -f $(APP_NAME)