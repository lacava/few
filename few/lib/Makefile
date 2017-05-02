.PHONY: all clean

all:
	g++ -fPIC -shared -O3 -I /usr/include/eigen3 -I /usr/include/python3.5 -std=c++0x epsilon_lexicase.cpp -lpython3.5m -o few_lib.so
	# g++ -std=c++0x -shared epsilon_lexicase.o -lpython3.5m -o few_lib.so
clean:
	rm few_lib.so
	rm epsilon_lexicase.o
