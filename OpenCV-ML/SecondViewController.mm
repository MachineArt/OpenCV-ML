//
//  SecondViewController.m
//  OpenCV-ML
//
//  Created by Zied on 2016-01-15.
//  Copyright © 2016 Machine Art. All rights reserved.
//

#import "SecondViewController.h"
#import "UIImage+OpenCV.h"
#import "Non_LinearSVM.h"

using namespace cv;

@interface SecondViewController ()

@property (nonatomic, weak) IBOutlet UIImageView * imageView;

@end

@implementation SecondViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.imageView.image = [UIImage imageNamed:@"training_process.png"];
    [self openCV_ML_Non_Linear_SVM];
}

- (void)openCV_ML_Non_Linear_SVM {
    
    Non_LinearSVM * non_LinearSVM;
    non_LinearSVM = new Non_LinearSVM();
        
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    __block cv::Mat image = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    
    // Run the SVM process in a background thread, in order to not block the UI (during user interaction)
    dispatch_async(queue, ^{
        
        non_LinearSVM->main_svm(image);

        dispatch_async(dispatch_get_main_queue(), ^{
            self.imageView.image = [UIImage UIImageFromCVMat:image];
        });

    });
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
