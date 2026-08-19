CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm
SRCS = main.c utils.c algo_01_to_06.c algo_07_to_09.c algo_10_to_13.c algo_14_to_15.c
OBJS = $(SRCS:.c=.o)
TARGET = matmul_test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c matmul_algorithms.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
