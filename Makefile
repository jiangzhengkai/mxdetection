all:
	cd mxdetection/ops/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../../
	cd mxdetection/ops/pycocotools; python setup.py build_ext --inplace; rm -rf build; cd ../../../
	cd mxdetection/ops/densecrf; make; cd ../../../
clean:
	cd mxdetection/ops/cython/; rm *.so *.c *.cpp; cd ../../../
	cd mxdetection/ops/pycocotools/; rm *.so _mask.c *.cpp; cd ../../../
	cd mxdetection/ops/densecrf; make; cd ../../../
