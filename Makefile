# Makefile

CC = gcc
CF = -O3 -fpic -Wall -pedantic
CL = -lm -lgsl -lgslcblas -fopenmp

all: cleft gs libgs

cleft: CLEFT.c
	$(CC) $(CF) CLEFT.c -o CLEFT $(CL)

gs: GaussianStreaming.c	
	$(CC) $(CF) GaussianStreaming.c -o GS $(CL)

libgs:
	$(CC) $(CF) -shared GaussianStreaming.c -o libCLEFT.so $(CL)

clean:
	rm -f *~ *.o

purge:
	rm -f GS CLEFT libCLEFT.so *~ *.o
