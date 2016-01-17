//
//  UIImage+OpenCV.h
//  OpenCV-ML
//
//  Created by Zied on 2016-01-15.
//  Copyright © 2016 Machine Art. All rights reserved.
//

const int WIDTH = 512, HEIGHT = 512;

@interface UIImage (OpenCV)

#pragma mark Generate UIImage from cv::Mat
+(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat;

#pragma mark Generate cv::Mat from UIImage
+ (cv::Mat)cvMatFromUIImage:(UIImage*)image;

@end