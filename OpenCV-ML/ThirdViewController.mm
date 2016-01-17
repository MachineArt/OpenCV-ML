//
//  ThirdViewController.m
//  OpenCV-ML
//
//  Created by Zied on 2016-01-15.
//  Copyright © 2016 Machine Art. All rights reserved.
//

#import "ThirdViewController.h"
#import "UIImage+OpenCV.h"
#import "PCA_Analysis.h"

using namespace cv;

@interface ThirdViewController ()

@property (nonatomic, weak) IBOutlet UIImageView * imageView;

@end

@implementation ThirdViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    [self openCV_ML_PCA];
}

- (void)openCV_ML_PCA {

    PCA_Analysis * pca_Analysis;
    pca_Analysis = new PCA_Analysis();
    
    cv::Mat image;
    image = [UIImage cvMatFromUIImage:[UIImage imageNamed:@"pca_test.jpg"]];
    
    pca_Analysis->main_pca(image);
    
    self.imageView.image = [UIImage UIImageFromCVMat:image];
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
