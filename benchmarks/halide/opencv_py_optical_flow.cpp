// Source:
// https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-01.cpp 

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

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
    int win_size = 128;

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
        cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
        1,                            // Maximum pyramid level to construct
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
            20,                         // Maximum number of iterations
            0.3                         // Minimum change per iteration
        )
    );

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end1 - start1;
    std::cout << "Time: " << duration.count() << std::endl;

    return 0;
}
