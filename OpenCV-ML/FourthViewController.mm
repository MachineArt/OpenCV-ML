//
//  FourthViewController.m
//  OpenCV-ML
//
//  Created by Zied on 2016-01-15.
//  Copyright © 2016 Machine Art. All rights reserved.
//

#import "FourthViewController.h"
#import "UIImage+OpenCV.h"
#import "Flat_Clustering.h"

using namespace cv;

@interface FourthViewController ()

@property (nonatomic, weak) IBOutlet UIImageView * imageView;

@end

@implementation FourthViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    [self openCV_ML_Flat_Clustering];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)openCV_ML_Flat_Clustering {

    Flat_Clustering * flat_Clustering;
    flat_Clustering = new Flat_Clustering();
    
    // Allocate space in the caller function
    cv::Mat image(500, 500, CV_8UC3);
    
    for (int iteration = 0; iteration < 1; iteration++) {
        flat_Clustering->main_kmeans(image);
        self.imageView.image = [UIImage UIImageFromCVMat:image];
    }
}

@end
