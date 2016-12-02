#ifndef H_TRANSMAT
#define H_TRANSMAT

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <Singleton.h>

using namespace cv;
using namespace std;

class TransMat : public Singleton<TransMat>
{
	friend class Singleton<TransMat>;

	private :
		TransMat();
		~TransMat();
		
		double rgb2yCbCr_data[3][3];
		double yCbCr2rgb_data[3][3];

		double rgb2lms_data[3][3];
		double lms2lab_data1[3][3];
		double lms2lab_data2[3][3];
		double lab2lms_data2[3][3];
		double lms2rgb_data[3][3];

		double min_rgb_data[3];
		double max_rgb_data[3];

	public : 		
		Mat min_yCbCr, max_yCbCr;

		Mat rgb2lms;		
		Mat lms2lab;
		Mat lab2lms;
		Mat lms2rgb;
		
		Mat rgb2yCbCr;
		Mat yCbCr2rgb;

		Mat image_rgb2lab(const Mat& im_rgb);//, Vec3d &lab_moy);
		Mat image_lab2rgb(const Mat& im_lab);
		
		Mat image_rgb2yCbCr(const Mat& im_rgb);
		Mat image_yCbCr2rgb(const Mat& im_rgb);
};

#endif