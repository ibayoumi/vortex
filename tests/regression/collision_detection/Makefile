ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := collision_detection

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/trace_loader.cpp $(SRC_DIR)/geometry.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n16

KERNEL_LIB := vortex2

include ../common.mk
