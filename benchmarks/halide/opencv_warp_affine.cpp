#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main( int, char** argv )
{
  Point2f srcTri[3];
  Point2f dstTri[3];
  Mat rot_mat( 2, 3, CV_32FC1 );
  Mat warp_mat( 2, 3, CV_32FC1 );
  Mat src0, src, warp_dst, warp_rotate_dst;

  src0 = imread( argv[1], 1 );
  cv::cvtColor(src0, src, CV_BGR2GRAY);
  warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
  srcTri[0] = Point2f( 0,0 );
  srcTri[1] = Point2f( src.cols - 1.f, 0 );
  srcTri[2] = Point2f( 0, src.rows - 1.f );
  dstTri[0] = Point2f( src.cols*0.0f, src.rows*0.33f );
  dstTri[1] = Point2f( src.cols*0.85f, src.rows*0.25f );
  dstTri[2] = Point2f( src.cols*0.15f, src.rows*0.7f );
  warp_mat = getAffineTransform( srcTri, dstTri );

  auto start1 = std::chrono::high_resolution_clock::now();
  warpAffine( src, warp_dst, warp_mat, warp_dst.size() );
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double,std::milli> duration = end1 - start1;
  std::cout << "Time: " << duration.count() << std::endl;

  Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
  double angle = -50.0;
  double scale = 0.6;
  rot_mat = getRotationMatrix2D( center, angle, scale );
  warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );
  return 0;
}
