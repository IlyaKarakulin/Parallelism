CC = g++
CFLAGS = -O2 -Wall -std=c++17
TYPE_FLAG =

ifdef USE_DOUBLE
    TYPE_FLAG = -DUSE_DOUBLE
endif

all:
	$(CC) $(CFLAGS) $(TYPE_FLAG) main.cpp -o program

clean:
	rm -f program
