//
//  FirstViewController.m
//  OpenCV-ML
//
//  Created by Zied on 2016-01-15.
//  Copyright © 2016 Machine Art. All rights reserved.
//

#import "FirstViewController.h"
#import "UIImage+OpenCV.h"
#import "LinearSVM.h"

using namespace cv;

@interface FirstViewController ()

@property (nonatomic, weak) IBOutlet UIImageView * imageView;

@end

@implementation FirstViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    [self openCV_ML_Linear_SVM];
}

- (void)openCV_ML_Linear_SVM {
    
    LinearSVM * linearSVM;
    linearSVM = new LinearSVM();
    
    // Allocate space in the caller function
    Mat image = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    linearSVM->main_svm(image);
    
    self.imageView.image = [UIImage UIImageFromCVMat:image];    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
