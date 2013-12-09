all: app.cu cudalist.cuh
	nvcc -O3 -arch=sm_30 app.cu -o benchmark

clean:
	rm -f benchmark
