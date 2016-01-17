
#include "LinearSVM.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

LinearSVM::LinearSVM()
{
    
}

void LinearSVM::main_svm(cv::Mat& image)
{
    // Set up training data
    //! [setup1]
    int labels[4] = {1, -1, -1, -1};
    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    //! [setup1]
    //! [setup2]
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);
    //! [setup2]
    
    
    // Train the SVM
    //! [init]
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    //! [init]
    //! [train]
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    //! [train]
    
    // Show the decision regions given by the SVM
    //! [show]
    Vec3b green(0,255,0), blue (0,0,255);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    //! [show]
    
    // Show the training data
    //! [show_data]
    int thickness = -1;
    int lineType = 8;
    circle(	image, cv::Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType ); // black color
    circle(	image, cv::Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType ); // white color
    circle(	image, cv::Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle(	image, cv::Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
    //! [show_data]
    
    // Show support vectors
    //! [show_vectors]
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getUncompressedSupportVectors(); // This needs OpenCV version 3.1!
    
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle(	image,  cv::Point( (int) v[0], (int) v[1]),   6,  Scalar(255, 0, 0), thickness, lineType); // red color
    }
    //! [show_vectors]
}