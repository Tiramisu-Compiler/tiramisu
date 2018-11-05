OPENCV_PREFIX=/Users/b/Documents/src-not-saved/opencv-3.2.0_prefix/

g++ --std=c++11 opencv_py_optical_flow.cpp -L${OPENCV_PREFIX}/lib/ -I${OPENCV_PREFIX}/include -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_imgcodecs -o opencv_py_optical_flow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OPENCV_PREFIX}/lib/

./opencv_py_optical_flow  ../../utils/images/rgb.png  ../../utils/images/rgb.png
