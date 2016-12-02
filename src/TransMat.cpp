#include "TransMat.h"

TransMat::TransMat()
{
	// RGB -> LMS
	//0.3811, 0.5783, 0.0402,
	//0.1967, 0.7244, 0.0782,
	//0.0241, 0.1288, 0.8444
	rgb2lms_data[0][0] = 0.3811;
	rgb2lms_data[0][1] = 0.5783;
	rgb2lms_data[0][2] = 0.0402;
	rgb2lms_data[1][0] = 0.1967;
	rgb2lms_data[1][1] = 0.7244;
	rgb2lms_data[1][2] = 0.0782;
	rgb2lms_data[2][0] = 0.0241;
	rgb2lms_data[2][1] = 0.1288;
	rgb2lms_data[2][2] = 0.8444;

	rgb2lms = Mat(3, 3, CV_64FC1, rgb2lms_data);
	
	// LMS -> LAB
	//1,	 1,	 1,
	//1,	 1,	-2,
	//1,	-1,  0
	lms2lab_data1[0][0] = 1;
	lms2lab_data1[0][1] = 1;
	lms2lab_data1[0][2] = 1;
	lms2lab_data1[1][0] = 1;
	lms2lab_data1[1][1] = 1;
	lms2lab_data1[1][2] = -2;
	lms2lab_data1[2][0] = 1;
	lms2lab_data1[2][1] = -1;
	lms2lab_data1[2][2] = 0;

	Mat lms2lab_1(3, 3, CV_64FC1, lms2lab_data1);

	//1.0/sqrt(3.0),	0,				0,
	//0,				1.0/sqrt(6.0),	0,
	//0,				0,				1.0/sqrt(2.0)
	lms2lab_data2[0][0] = 1.0/sqrt(3.0);
	lms2lab_data2[0][1] = 0;
	lms2lab_data2[0][2] = 0;
	lms2lab_data2[1][0] = 0;
	lms2lab_data2[1][1] = 1.0/sqrt(6.0);
	lms2lab_data2[1][2] = 0;
	lms2lab_data2[2][0] = 0;
	lms2lab_data2[2][1] = 0;
	lms2lab_data2[2][2] = 1.0/sqrt(2.0);

	Mat lms2lab_2(3, 3, CV_64FC1, lms2lab_data2);
	lms2lab = lms2lab_1.t() * lms2lab_2;
	
	// LAB -> LMS
	//sqrt(3.0)/3.0,	0,				0,
	//0,				sqrt(6.0)/6.0,	0,
	//0,				0,				sqrt(2.0)/2.0
	lab2lms_data2[0][0] = sqrt(3.0)/3.0;
	lab2lms_data2[0][1] = 0;
	lab2lms_data2[0][2] = 0;
	lab2lms_data2[1][0] = 0;
	lab2lms_data2[1][1] = sqrt(6.0)/6.0;
	lab2lms_data2[1][2] = 0;
	lab2lms_data2[2][0] = 0;
	lab2lms_data2[2][1] = 0;
	lab2lms_data2[2][2] = sqrt(2.0)/2.0;

	Mat lab2lms_2_2(3, 3, CV_64FC1, lab2lms_data2);
	lab2lms = lms2lab_1.t() * lab2lms_2_2;

	
	// LMS -> RGB
	//4.4679, -3.5873, 0.1193,
	//-1.2186, 2.3809, -0.1624,
	//0.0497, -0.2439, 1.2045
	lms2rgb_data[0][0] = 4.4679;
	lms2rgb_data[0][1] = -3.5873;
	lms2rgb_data[0][2] = 0.1193;
	lms2rgb_data[1][0] = -1.2186;
	lms2rgb_data[1][1] = 2.3809;
	lms2rgb_data[1][2] = -0.1624;
	lms2rgb_data[2][0] = 0.04970;
	lms2rgb_data[2][1] = -0.2439;
	lms2rgb_data[2][2] = 1.2045;

	lms2rgb = Mat(3, 3, CV_64FC1, lms2rgb_data);

	
	rgb2yCbCr_data[0][0] = 0.2990;
	rgb2yCbCr_data[0][1] = 0.5870;
	rgb2yCbCr_data[0][2] = 0.1140;
	rgb2yCbCr_data[1][0] = -0.1687;
	rgb2yCbCr_data[1][1] = -0.3313;
	rgb2yCbCr_data[1][2] = 0.5000;
	rgb2yCbCr_data[2][0] = 0.5000;
	rgb2yCbCr_data[2][1] = -0.4187;
	rgb2yCbCr_data[2][2] = -0.0813;
	rgb2yCbCr = Mat(3, 3, CV_64FC1, rgb2yCbCr_data);
	
	yCbCr2rgb_data[0][0] = 1;
	yCbCr2rgb_data[0][1] = 0;
	yCbCr2rgb_data[0][2] = 1.4020;
	yCbCr2rgb_data[1][0] = 1;
	yCbCr2rgb_data[1][1] = -0.3441;
	yCbCr2rgb_data[1][2] = -0.7141;
	yCbCr2rgb_data[2][0] = 1;
	yCbCr2rgb_data[2][1] = 1.7720;
	yCbCr2rgb_data[2][2] = 0;
	yCbCr2rgb = Mat(3, 3, CV_64FC1, yCbCr2rgb_data);

	
	min_rgb_data[0] = 0;
	min_rgb_data[1]	= 0;
	min_rgb_data[2]	= 0;
	max_rgb_data[0]	= 255;
	max_rgb_data[1]	= 255;
	max_rgb_data[2]	= 255;

	//Mat min_rgb(1,1, CV_8UC3, min_rgb_data), max_rgb(1,1, CV_8UC3, max_rgb_data);
	min_yCbCr = rgb2yCbCr * Mat(3,1, CV_64FC1, min_rgb_data);//image_rgb2yCbCr(min_rgb);
	max_yCbCr = rgb2yCbCr * Mat(3,1, CV_64FC1, max_rgb_data);//image_rgb2yCbCr(max_rgb);
}

Mat TransMat::image_rgb2yCbCr(const Mat& im_rgb)
{
	Mat im_yCbCr = Mat::zeros(im_rgb.size(), CV_64FC3);
		
	// for each (pixel in imageSource)
	MatConstIterator_<Vec3b> it = im_rgb.begin<Vec3b>();
	MatIterator_<Vec3d> it_yCbCr = im_yCbCr.begin<Vec3d>();
	
	for(int i = 0; it != im_rgb.end<Vec3b>(); ++it, ++it_yCbCr, ++i)
	{
		Vec3d vTmp = (*it);
		double data[3]  = {vTmp[0], vTmp[1], vTmp[2]};

		//Mat pixel = TransMat::instance().rgb2yCbCr;
		Mat pixel = TransMat::instance().rgb2yCbCr * Mat(3,1, CV_64FC1, data);
		Vec3d pValue = pixel;

		(*it_yCbCr) = pValue;
	}

	return im_yCbCr;
}

Mat TransMat::image_yCbCr2rgb(const Mat& im_yCbCr)
{
	Mat im_rgb = Mat::zeros(im_yCbCr.size(), CV_64FC3);//CV_8UC3);

	MatConstIterator_<Vec3d> it_yCbCr = im_yCbCr.begin<Vec3d>();
	MatIterator_<Vec3d> it_rgb = im_rgb.begin<Vec3d>();
	for(; it_yCbCr != im_yCbCr.end<Vec3d>(); ++it_yCbCr, ++it_rgb)
	{
		Vec3d test = (*it_yCbCr);
		double data[3][1]  = {test[0], test[1], test[2]};

		Mat pixel = TransMat::instance().yCbCr2rgb * Mat(3,1, CV_64FC1, data);

		*it_rgb = pixel;
	}

	return im_rgb;
}


Mat TransMat::image_rgb2lab(const Mat& im_rgb)//, Vec3d &lab_moy)
{
	Mat im_lab = Mat::zeros(im_rgb.size(), CV_64FC3);
		
	// for each (pixel in imageSource)
	MatConstIterator_<Vec3b> it = im_rgb.begin<Vec3b>();
	MatIterator_<Vec3d> it_lab = im_lab.begin<Vec3d>();
	
	for(int i = 0; it != im_rgb.end<Vec3b>(); ++it, ++it_lab, ++i)
	{
		Vec3d vTmp = (*it);
		double data[3]  = {vTmp[0], vTmp[1], vTmp[2]};

		Mat tmp = TransMat::instance().rgb2lms * Mat(3,1, CV_64FC1, data);
		
		if(tmp.at<double>(0) == 0.0)
			tmp.at<double>(0) = 0.1;

		if(tmp.at<double>(1) == 0.0)
			tmp.at<double>(1) = 0.1;

		if(tmp.at<double>(2) == 0.0)
			tmp.at<double>(2) = 0.1;

		
		/*if(tmp.at<double>(0) != 0.0)
			tmp.at<double>(0) = log10(tmp.at<double>(0));

		if(tmp.at<double>(1) != 0.0)
			tmp.at<double>(1) = log10(tmp.at<double>(1));

		if(tmp.at<double>(2) != 0.0)
			tmp.at<double>(2) = log10(tmp.at<double>(2));*/

		
		tmp.at<double>(0) = log10(tmp.at<double>(0));
		tmp.at<double>(1) = log10(tmp.at<double>(1));
		tmp.at<double>(2) = log10(tmp.at<double>(2));
		
		Mat pixel = TransMat::instance().lms2lab.t() * tmp;
		Vec3d pValue = pixel;

		(*it_lab) = pValue;
	}

	return im_lab;
}

Mat TransMat::image_lab2rgb(const Mat& im_lab)
{
	Mat im_rgb = Mat::zeros(im_lab.size(), CV_8UC3);

	MatConstIterator_<Vec3d> it_lab = im_lab.begin<Vec3d>();
	MatIterator_<Vec3b> it_rgb = im_rgb.begin<Vec3b>();
	for(; it_lab != im_lab.end<Vec3d>(); ++it_lab, ++it_rgb)
	{
		Vec3d test = (*it_lab);
		double data[3][1]  = {test[0], test[1], test[2]};

		Mat tmp = TransMat::instance().lab2lms * Mat(3,1, CV_64FC1, data);

		tmp.at<double>(0) = pow(10,tmp.at<double>(0));
		tmp.at<double>(1) = pow(10,tmp.at<double>(1));
		tmp.at<double>(2) = pow(10,tmp.at<double>(2));

		Mat pixel = TransMat::instance().lms2rgb * tmp;

		*it_rgb = pixel;
	}

	return im_rgb;
}