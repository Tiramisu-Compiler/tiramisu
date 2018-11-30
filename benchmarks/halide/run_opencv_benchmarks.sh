OPENCV_PREFIX=/Volumes/ALL/extra/opencv-3.2.0_prefix/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OPENCV_PREFIX}/lib/


g++ --std=c++11 opencv_py_optical_flow.cpp -L${OPENCV_PREFIX}/lib/ -I${OPENCV_PREFIX}/include -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_imgcodecs -o opencv_py_optical_flow
g++ --std=c++11 opencv_warp_affine.cpp -L${OPENCV_PREFIX}/lib/ -I${OPENCV_PREFIX}/include -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_imgcodecs -o opencv_warp_affine

echo "Running LK Optical Flow"
./opencv_py_optical_flow  ../../utils/images/gray.png  ../../utils/images/gray.png

echo "Running warpAffine"
./opencv_warp_affine ../../utils/images/rgb.png
