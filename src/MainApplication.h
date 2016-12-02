#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <TransMat.h>

void colorTransfert_image2Vid(Mat &imageSource, VideoCapture &videoTarget, int nbFrames);
void colorTransfert_vid2Vid(VideoCapture &videoSource, VideoCapture &videoTarget, int nbFrames);
Mat  colorTransfert_image2image(Mat &imageSource, Mat &imageTarget);
void swatchesColorTransfert_image2image(Mat imageSource, Mat imageTarget, std::vector<Point> pointsSource, std::vector<Point> pointsTarget);
void mouseHandlerSource(int event, int x, int y, int flags, void* param);
void mouseHandlerTarget(int event, int x, int y, int flags, void* param);
void computePowerMatrix(Mat &covI, Mat &SigmaIHalfPow, float exposant);
void histo(Mat & srcLab,Mat & colorizedLab, Mat & targetLab, int colorType);
Mat createMaskAlpha(Mat & ImageTarget, int direction);