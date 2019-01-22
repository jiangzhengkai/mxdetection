CXX := g++
CFLAGS := -W -Wall -O2 -Wunused-parameter

SRCS_CPP := c_api.cpp permutohedral_c.cpp
SRCS_O := c_api.o permutohedral_c.o

all:
	make permutohedral_c.a
	make permutohedral_c.so
	
clean:
	rm -f *.o
	rm -f *.a
	rm -f *.so
	

permutohedral_c.a: $(SRCS_CPP)
	$(CXX) $(SRCS_CPP) -c $(CFLAGS)
	ar cr permutohedral_c.a $(SRCS_O)
	rm -f *.o
	
	
permutohedral_c.so: $(SRCS_CPP)
	$(CXX) $(SRCS_CPP) -c -fPIC $(CFLAGS)
	$(CXX) -shared -fPIC -o permutohedral_c.so $(SRCS_O)
	rm -f *.o
	