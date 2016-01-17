
#include "Flat_Clustering.h"

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

 static void help()
 {
     cout<< "\n--------------------------------------------------------------------------" << endl
         << "This program shows Unsupervised Machine Learning: Flat Clustering. " << endl
         << "\nThis program demonstrates kmeans clustering.\n"
             "It generates an image with random points, then assigns a random number of cluster "
             "centers and uses kmeans to move those cluster centers to their representitive location\n"
             "Class:\n"
             "./Flat_Clustering\n" << endl;
 }

Flat_Clustering::Flat_Clustering()
{

}

void Flat_Clustering::main_kmeans(cv::Mat& img)
{
    help();
    
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
    
    RNG rng(12345);
    
    int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
    int i, sampleCount = rng.uniform(1, 1001);
    Mat points(sampleCount, 1, CV_32FC2), labels;
    
    clusterCount = MIN(clusterCount, sampleCount);
    Mat centers;
    
    /* generate random sample from multigaussian distribution */
    for( k = 0; k < clusterCount; k++ )
    {
        Point center;
        center.x = rng.uniform(0, img.cols);
        center.y = rng.uniform(0, img.rows);
        Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                         k == clusterCount - 1 ? sampleCount :
                                         (k+1)*sampleCount/clusterCount);
        rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
    }
    
    randShuffle(points, 1, &rng);

    kmeans(points, clusterCount, labels,
           TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
           3, KMEANS_PP_CENTERS, centers);
    
    img = Scalar::all(0);
    
    cout<< "clusterCount == " << clusterCount << endl;
    cout<< "sampleCount == " << sampleCount << endl;
//    cout<< "points == " << points << endl;
    cout<< "centers == " << centers << endl;
    
        for( i = 0; i < sampleCount; i++ )
        {
            int clusterIdx = labels.at<int>(i);
            Point ipt = points.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
        }

    for( i = 0; i < clusterCount; i++ )
    {
        circle( img, centers.at<Point2f>(i), 8, colorTab[4], FILLED, LINE_AA );
    }

}
