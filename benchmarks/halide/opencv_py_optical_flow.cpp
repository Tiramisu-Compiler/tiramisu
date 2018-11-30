// Source:
// https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-01.cpp 

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "wrapper_py_optical_flow_sizes.h"

static const int MAX_CORNERS = 8;
using std::cout;
using std::endl;
using std::vector;


void help( char** argv ) {
  cout << "\nExample 16-1: Pyramid L-K optical flow example.\n" << endl;
  cout << "Call: " <<argv[0] <<" [image1] [image2]\n" << endl;
  cout << "\nExample:\n" << argv[0] << " ../example_16-01-imgA.png ../example_16-01-imgB.png\n" << endl;
  cout << "Demonstrates Pyramid Lucas-Kanade optical flow.\n" << endl;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        help(argv);
        exit(-1);
    }

    cv::Mat imgA = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat imgB = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Size img_sz = imgA.size();
    int win_size = w;
    cv::Size sz(win_size * 2 + 1, win_size * 2 + 1);

    vector< cv::Point2f > cornersA, cornersB;

    cornersA.push_back(cv::Point2f(500, 400));
    cornersA.push_back(cv::Point2f(800, 900));
    cornersA.push_back(cv::Point2f(200, 400));
    cornersA.push_back(cv::Point2f(400, 200));
    cornersA.push_back(cv::Point2f(400, 500));
    cornersA.push_back(cv::Point2f(800, 200));
    cornersA.push_back(cv::Point2f(200, 900));
    cornersA.push_back(cv::Point2f(900, 200));

    // Call the Lucas Kanade algorithm
    vector<uchar> features_found;

    auto start1 = std::chrono::high_resolution_clock::now();

    cv::calcOpticalFlowPyrLK(
        imgA,                         // Previous image
        imgB,                         // Next image
        cornersA,                     // Previous set of corners (from imgA)
        cornersB,                     // Next set of corners (from imgB)
        features_found,               // Output vector, each is 1 for tracked
        cv::noArray(),                // Output vector, lists errors (optional)
        sz,  // Search window size
        0,                            // Maximum pyramid level to construct
        cv::TermCriteria(
            cv::TermCriteria::COUNT,
            1,                         // Maximum number of iterations
            0.3                         // Minimum change per iteration
        )
    );

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end1 - start1;
    std::cout << "Sparse 0-Pyramid Lucas-Kanade Time: " << duration.count() << std::endl;


    // ----------------------------------------------------------------
    // ----------------------------------------------------------------
    // ----------------------------------------------------------------
    // ----------------------------------------------------------------


    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> prevPyr, nextPyr;

    cv::Mat nimgA(cv::Size(PY_IMG_SIZE, PY_IMG_SIZE), imgA.type());
    cv::Mat nimgB(cv::Size(PY_IMG_SIZE, PY_IMG_SIZE), imgB.type());

    buildOpticalFlowPyramid(nimgA, prevPyr, sz, npyramids, true);
    buildOpticalFlowPyramid(nimgB, nextPyr, sz, npyramids, true);

    for (int i = 0; i < nimgA.cols; i++)
	for (int j = 0; j < nimgA.rows; j++)
	{
	    vector< cv::Point2f > cornersA0, cornersB0;

	    cornersA0.push_back(cv::Point2f(j, i));

	    cv::calcOpticalFlowPyrLK(
		prevPyr,			// Previous image
		nextPyr,                        // Next image
		cornersA0,                      // Previous set of corners (from nimgA)
		cornersB0,                      // Next set of corners (from imgB)
		features_found,                 // Output vector, each is 1 for tracked
		cv::noArray(),                  // Output vector, lists errors (optional)
		sz,				// Search window size
		npyramids,                      // Maximum pyramid level to construct
		cv::TermCriteria(
		    cv::TermCriteria::COUNT,
		    niterations,                          // Maximum number of iterations
		    0.3                         // Minimum change per iteration
		)
	    );
    }

    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
    std::cout << "Dense Pyramidal Lucas-Kanade Time: " << duration2.count() << std::endl;

    return 0;
}
