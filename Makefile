# Makefile for compiling the CLPT_GS code on Linux systems

CC = gcc
CF = -O3 -Wall -Wextra -Wno-unused-parameter -Wuninitialized -Winit-self -pedantic -ffast-math -std=gnu11

LIBRARIES = -lgsl -lgslcblas -lm -fopenmp

#FINALIZE COMPILE FLAGS
CF += $(OPTIONS) #-g

## FINALIZE 
CLPT_INC = $(INCLUDES)
CLPT_LIB =  $(LIBRARIES) 

all: CLEFT GS

CLEFT: Code_RSD_CLEFT.c 
	$(CC) Code_RSD_CLEFT.c -o CLEFT $(CF) $(CLPT_INC) $(CLPT_LIB)

GS: Code_RSD_GS.c
	$(CC) Code_RSD_GS.c -o GS $(CF) $(CLPT_INC) $(CLPT_LIB)

clean:
	rm -f CLEFT GS */*~ *~

lib:    
	$(CC) -shared Code_RSD_GS.c  -fpic $(CF) $(CLPT_INC) $(CLPT_LIB) -o libGSCLEFT.so
