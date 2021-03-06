#include "MainApplication.h"

using namespace cv;
using namespace std;
//
//#define nb_swatches 1
//#define swatch_width 600
//#define swatch_height 450
//#define nb_swatches 3
//#define swatch_width 10
//#define swatch_height 10

int nb_swatches;
int swatch_width;
int swatch_height;
Mat imSourceDisplayed;
Mat imTargetDisplayed; 

std::vector<Scalar> colors;

int main( int argc, char* argv[])
{
	std::string transferType = "img2img"; // img2img, img2vid, vid2vid
	std::string sourceFile = "data/vangogh.jpg";
	std::string targetFile = "data/Capture.PNG";

	if(argc >= 4)
	{
		transferType	= argv[1];
		sourceFile		= argv[2];
		targetFile		= argv[3];
	}

	if(transferType == "img2img")
	{
		/*
		std::string sourceFile = "data/piscine.jpg";
		std::string targetFile = "data/foot.jpg";
		*/

		Mat imageSource, imageTarget;
		imageSource = imread(sourceFile, CV_LOAD_IMAGE_COLOR);	 // Read the file
		imageTarget = imread(targetFile, CV_LOAD_IMAGE_COLOR);   // Read the file

		if(! imageSource.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the source image" << std::endl ;
			return -1;
		}
		if(! imageTarget.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the target image" << std::endl ;
			return -1;
		}

		imSourceDisplayed = imageSource.clone();
		imTargetDisplayed = imageTarget.clone();

		namedWindow( "Color image source", CV_WINDOW_AUTOSIZE );// Create a window for display.
		imshow( "Color image source", imSourceDisplayed ); 
		namedWindow( "Color image target", CV_WINDOW_AUTOSIZE );// Create a window for display.
		imshow( "Color image target", imTargetDisplayed ); 
		//bool useSwatches = false;

		std::cout<<"Entrez le nombre de swatches voulus (0 pour utiliser la methode globale) :"<<std::endl;
		std::cin>>nb_swatches;
		if(nb_swatches > 0)
		{
			std::cout<<"Entrez la largeur des swatches :"<<std::endl;
			std::cin>>swatch_width;
			std::cout<<"Entrez la hauteur des swatches :"<<std::endl;
			std::cin>>swatch_height;
	
			std::vector<Point> *pointsSource = new std::vector<Point>;
			std::vector<Point> *pointsTarget = new std::vector<Point>;
				("Color image source", mouseHandlerSource, pointsSource);
			setMouseCallback("Color image target", mouseHandlerTarget, pointsTarget);
	
			colors.push_back(Scalar(0,0,255));
			colors.push_back(Scalar(0,255,0));
			colors.push_back(Scalar(255,0,0));
			colors.push_back(Scalar(0,128,128));
			colors.push_back(Scalar(128,128,0));
			colors.push_back(Scalar(128,0,128));
			do
			{
				waitKey(0);
				std::cout<<pointsSource->size()<<" et "<<pointsTarget->size()<<std::endl;
			}while(pointsSource->size() != nb_swatches || pointsTarget->size() != nb_swatches);

			imwrite("results/swatches/swatch_source.png", imSourceDisplayed);
			imwrite("results/swatches/swatch_target.png", imTargetDisplayed);


			swatchesColorTransfert_image2image(imageSource, imageTarget, *pointsSource, *pointsTarget);
		}
		else
			colorTransfert_image2image(imageSource, imageTarget);
	}
	else if(transferType == "vid2vid") // img2vid
	{
		VideoCapture videoSource(sourceFile);
		VideoCapture videoTarget(targetFile);
		
		if(!videoSource.isOpened())
		{
			cout <<  "Could not open or find the source video" << std::endl ;
			return -1;
		}
		if(!videoTarget.isOpened())
		{
			cout <<  "Could not open or find the target video" << std::endl ;
			return -1;
		}
		
		int nbFrames ;
		std::cout<<"Entrez le nombre de frames a traiter : "<<std::endl;
		std::cin>>nbFrames;
		colorTransfert_vid2Vid(videoSource, videoTarget, nbFrames);
	}
	else if(transferType == "img2vid") //vid2vid
	{
		VideoCapture videoTarget(targetFile);
		Mat imageSource;
		imageSource = imread(sourceFile, CV_LOAD_IMAGE_COLOR);	 // Read the file

		if(! imageSource.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the source image" << std::endl ;
			return -1;
		}
		if(!videoTarget.isOpened())
		{
			cout <<  "Could not open or find the target video" << std::endl ;
			return -1;
		}
		
		int nbFrames = 15*25;
		std::cout<<"Entrez le nombre de frames a traiter : "<<std::endl;
		std::cin>>nbFrames;
		colorTransfert_image2Vid(imageSource, videoTarget, nbFrames);
	}

	waitKey(0);
	return 0;
}

void colorTransfert_image2Vid(Mat &imageSource, VideoCapture &videoTarget, int nbFrames)
{
	VideoWriter *videoResult = NULL;
    for(int i = 0; i < nbFrames; ++i)
	{
        Mat frame;
        videoTarget >> frame; // get a new frame from camera
		
		if(videoResult == NULL)
			videoResult = new VideoWriter("results/result.avi", -1, 25, frame.size(), true);
		
		Mat frameColorized = colorTransfert_image2image(imageSource,frame);
		//normalize(frame, frame, 0, 255, NORM_MINMAX);

		(*videoResult) << frameColorized;

	}

	cout <<  "Finish." << std::endl ;

	//TransMat::deleteInstance();
	delete videoResult;
    waitKey(0);                                          // Wait for a keystroke in the window
}

void colorTransfert_vid2Vid(VideoCapture &videoSource, VideoCapture &videoTarget, int nbFrames)
{
	VideoWriter *videoResult = NULL;

	Mat pre_frameSource;
	videoSource >> pre_frameSource; 	

	Mat frameSource;
    videoSource >> frameSource; // get a new frame from camera

	Mat post_frameSource;
       
    for(int i = 1; i < nbFrames; ++i)
	{
        videoSource >> post_frameSource; // get a new frame from camera
		Mat frameTarget;
        videoTarget >> frameTarget; // get a new frame from camera
		
		if(videoResult == NULL)
			videoResult = new VideoWriter("results/result.avi", -1, 25, frameTarget.size(), true);
		

		Mat frameColorized = colorTransfert_image2image(post_frameSource,frameTarget);
		//normalize(frame, frame, 0, 255, NORM_MINMAX);
		namedWindow( "Image Result", CV_WINDOW_AUTOSIZE ); // Create a window for display.
		imshow( "Image Result", frameColorized );             // Show our image inside it.
		
		std::cout<<i<<" / "<<nbFrames<<std::endl;
		(*videoResult) << frameColorized;

	}

	cout <<  "Finish." << std::endl ;

	//TransMat::deleteInstance();
	delete videoResult;
    waitKey(0);                                          // Wait for a keystroke in the window
}

void computeTransformMatrix(Mat &ImT_lab, Mat &ImS_lab, Mat &MuTarget, Mat &MuSource,Mat & Transform)
{
	//Calcul de la moyenne et �cart-type de l'image Target
	Scalar labMoy_t, stddev_t;
	meanStdDev(ImT_lab, labMoy_t, stddev_t);

	//Calcul de la moyenne et �cart-type de l'image source
	Scalar labMoy_s, stddev_s;
	meanStdDev(ImS_lab, labMoy_s, stddev_s);
	// CALCUL DE LA MATRICE SAMPLE S QUI VA PERMETTRE DE CALCULER LA MATRICE DE COV SIGMAs
	Mat sampleS = Mat::zeros(ImS_lab.rows*ImS_lab.cols, 2, CV_32FC1);
	Mat sampleI = Mat::zeros(ImT_lab.rows*ImT_lab.cols, 2, CV_32FC1);

	for (int i = 0; i < ImS_lab.rows; i++)
	{
		for (int j = 0; j < ImS_lab.cols; j++)
		{
			sampleS.at<float>(i*ImS_lab.cols + j, 0) = ImS_lab.at<Vec3f>(i, j)[1];  // a
			sampleS.at<float>(i*ImS_lab.cols + j, 1) = ImS_lab.at<Vec3f>(i, j)[2]; // b
		}
	}
	// Calcul matrice de cov SigmaS
	Mat covS, muS;
	cv::calcCovarMatrix(sampleS, covS, muS, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32FC1);

	covS = covS / (sampleS.rows - 1);

	// ----- //
	// CALCUL DE LA MATRICE SAMPLE i QUI VA PERMETTRE DE CALCULER LA MATRICE DE COV SIGMAi


	for (int i = 0; i < ImT_lab.rows; i++)
	{
		for (int j = 0; j < ImT_lab.cols; j++)
		{
			sampleI.at<float>(i*ImT_lab.cols + j, 0) = ImT_lab.at<Vec3f>(i, j)[1];  // a
			sampleI.at<float>(i*ImT_lab.cols + j, 1) = ImT_lab.at<Vec3f>(i, j)[2]; // b
		}
	}
	// Calcul matrice de cov SigmaI
	Mat covI, muI;
	cv::calcCovarMatrix(sampleI, covI, muI, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32FC1);
	//?? que fais cette ligne que l'on a copier coller ??
	covI = covI / (sampleI.rows - 1);

	//R�gularisation : = max(covI, 7.5Identity)
	covI.at<float>(0, 0) = (covI.at<float>(0, 0) < 7.5f) ? 7.5f : covI.at<float>(0, 0);
	covI.at<float>(1, 1) = (covI.at<float>(1, 1) < 7.5f) ? 7.5f : covI.at<float>(1, 1);

	//Matrice permettant le calcul de la transform�e
	// Transform = SigmaIMinusHalfPow * ( SigmaIHalfPow * covS * SigmaIHalfPow)^1/2 * SigmaIMinusHalfPow

	Mat SigmaIMinusHalfPow = Mat::zeros(2, 2, CV_32FC1);
	Mat SigmaIHalfPow = Mat::zeros(2, 2, CV_32FC1);

	computePowerMatrix(covI, SigmaIHalfPow, 0.5f);
	computePowerMatrix(covI, SigmaIMinusHalfPow, -0.5f);

	// TransformInterSig = SigmaIHalfPow * covS * SigmaIHalfPow
	Mat TransformInterSig = Mat::zeros(2, 2, CV_32FC1);
	TransformInterSig = SigmaIHalfPow * covS * SigmaIHalfPow;

	// TransformInterSigHalfPow = TransformInterSig ^ 1/2
	Mat TransformInterSigHalfPow = Mat::zeros(2, 2, CV_32FC1);
	computePowerMatrix(TransformInterSig, TransformInterSigHalfPow, 0.5);

	//Calcul de la transform�e
	Transform = SigmaIMinusHalfPow * TransformInterSigHalfPow * SigmaIMinusHalfPow;

	// Matrice [MoyenneA,MoyennneB] avec MoyenneA la moyenne de a sur la target (vice versa b) 
	//Mat MuTarget = Mat::zeros(2,1, CV_32FC1);
	MuTarget.at<float>(0, 0) = labMoy_t[1];  // a
	MuTarget.at<float>(1, 0) = labMoy_t[2];  // b
	// Matrice [MoyenneA,MoyennneB] avec MoyenneA la moyenne de a sur la source (vice versa b) 

	//Mat MuSource = Mat::zeros(2,1, CV_32FC1);
	MuSource.at<float>(0, 0) = labMoy_s[1];  // a
	MuSource.at<float>(1, 0) = labMoy_s[2];  // b
}

Mat colorTransfert_image2image(Mat &imageSource, Mat &imageTarget)
{
	// Image source dans l'espace de couleurs Lab
	Mat ImS_lab;
	//matrice temporaire utilis�e pour la conversion
	Mat imageSourceConverted;
	imageSource.convertTo(imageSourceConverted,CV_32FC3);
	imageSourceConverted *= 1./255; // Normalisation

	//Conversion en BGR -> Lab
	cvtColor(imageSourceConverted,ImS_lab, CV_BGR2Lab);


		
	// Image Target dans l'espace de couleurs Lab
	Mat ImT_lab;
	Mat imageTargetConverted;
	imageTarget.convertTo(imageTargetConverted,CV_32FC3);
	imageTargetConverted *= 1./255; // Normalisation	
	//Conversion en Lab
	cvtColor(imageTargetConverted,ImT_lab, CV_BGR2Lab);
	
	Mat MuTarget = Mat::zeros(2, 1, CV_32FC1);
	Mat MuSource = Mat::zeros(2, 1, CV_32FC1);
	Mat Transform = Mat::zeros(2, 2, CV_32FC1);

	//Image resultat dans l'espace Lab
	Mat ImR_lab = Mat::zeros(ImT_lab.size(), CV_32FC3);

	computeTransformMatrix(ImT_lab, ImS_lab,MuTarget,MuSource,Transform);

	Mat MaskAlpha = createMaskAlpha(ImT_lab,2); //0 or 1(other sense) for horizontal | 2 or 3(other sense) for vertical 


	float alpha;
	//calcul de co(x) = T(ci(x)-MuTarget)+MuSource
	for(int i=0; i<imageTarget.rows;i++)
	{
		for(int j=0; j<imageTarget.cols;j++)
		{
			alpha = MaskAlpha.at<float>(i,j);

			Mat CTarget = Mat::zeros(2,1, CV_32FC1);
			CTarget.at<float>(0,0) = ImT_lab.at<Vec3f>(i,j)[1];  // a
			CTarget.at<float>(1,0) = ImT_lab.at<Vec3f>(i,j)[2];  // b

			Mat MatTrans = (Transform * (CTarget - MuTarget)) + MuSource;
			ImR_lab.at<Vec3f>(i,j)[0] = ImT_lab.at<Vec3f>(i,j)[0]; // L
			ImR_lab.at<Vec3f>(i,j)[1] = ImT_lab.at<Vec3f>(i,j)[1] *alpha +(1-alpha) * MatTrans.at<float>(0,0); // a
			ImR_lab.at<Vec3f>(i,j)[2] = ImT_lab.at<Vec3f>(i,j)[2] *alpha +(1-alpha) * MatTrans.at<float>(1,0); // b 
		}
	}
		
	// back to RGB
	Mat imageResult ;
	cvtColor(ImR_lab,imageResult, CV_Lab2BGR);
	//affichage
	namedWindow( "Color Result", CV_WINDOW_AUTOSIZE );
	imshow( "Color Result", imageResult );
	//Ecriture
	//normalize(imageResult, imageResult, 0, 255, NORM_MINMAX);
	//imwrite("results/img2img.png", imageResult);

	//Mat ImR_bgr;
	//cvtColor(ImR_lab,ImR_bgr, CV_Lab2BGR);
	//histo(imageSource, imageTarget , ImT_lab, 1);

	//temporaire pour video :

	return imageResult;
}

// ON EST ICIIIII � MECS SERIEUX LA sA Marche Votre TRUC !!!!

// Pour tester : Tracer histogramme a & b avant et apr�s
// => Il doit avoir (convergence| les histogramme doivent se rapprocher | plus coller sur la source)

// Grosse Etape : Gerer Les patch !


// Etape suivante la suivante : Image2Video ||Video2Video


Mat createMaskAlpha(Mat &ImageTarget, int direction)
{
	int height = ImageTarget.rows;
	int width = ImageTarget.cols;

	Mat MaskAlpha = Mat::zeros(height,width, CV_32FC1);
	if(direction == 5)
		return MaskAlpha;

	for(int i=0; i<height;i++)
		for(int j=0;j<width;j++)
			if(direction == 0)
				MaskAlpha.at<float>(i,j) = 1-((1/(float)width)*(float)j);
			else if(direction == 1)
				MaskAlpha.at<float>(i,j) = ((1/(float)width)*(float)j);
			else if(direction == 2)
				MaskAlpha.at<float>(i,j) = 1-((1/(float)height)*(float)i);
			else if(direction == 3)
				MaskAlpha.at<float>(i,j) = ((1/(float)height)*(float)i);


	return MaskAlpha;
}


void computePowerMatrix(Mat &covI, Mat &SigmaIHalfPow, float exposant)
{
	Mat eigenValue;
	Mat eigenVector = Mat::zeros(2,2, CV_32FC1);

	cv::eigen(covI,eigenValue,eigenVector);

	SigmaIHalfPow.at<float>(0,0) = pow(eigenValue.at<float>(0,0),exposant);
	SigmaIHalfPow.at<float>(1,1) = pow(eigenValue.at<float>(1,0),exposant);
}



void swatchesColorTransfert_image2image(Mat imageSource, Mat imageTarget, std::vector<Point> pointsSource, std::vector<Point> pointsTarget)
{

	//Changer en Lab et supprimer Transmat
	Mat ImS_Lab; 
	Mat ImT_Lab; 
	Mat imageSourceConverted;
	Mat imageTargetConverted;
	
	imageSource.convertTo(imageSourceConverted, CV_32FC3);
	imageSourceConverted *= 1. / 255; // Normalisation

	imageTarget.convertTo(imageTargetConverted, CV_32FC3);
	imageTargetConverted *= 1. / 255; // Normalisation

	 //Conversion en BGR -> Lab
	cvtColor(imageSourceConverted, ImS_Lab, CV_BGR2Lab);
	cvtColor(imageTargetConverted, ImT_Lab, CV_BGR2Lab);


	// computation of stddev and mean for each swatch in the source image
	std::vector<Scalar> stddevs_S, moys_S;
	for(int i = 0; i < nb_swatches; ++i)
	{
		Point topLeft = pointsSource[i]-Point(swatch_width/2, swatch_height/2);
		Mat swatch = ImS_Lab.colRange(topLeft.x,topLeft.x+swatch_width).rowRange(topLeft.y,topLeft.y+swatch_height); 
		Scalar swatchMoy, swatchstddev;
		meanStdDev(swatch, swatchMoy, swatchstddev);
		moys_S.push_back(swatchMoy);
		stddevs_S.push_back(swatchstddev);
		std::cout<<"MoysS :"<<swatchMoy<<std::endl;
		std::cout<<"stdS :"<<swatchstddev<<std::endl;
	}
	
	// computation of stddev and mean for each swatch in the target image
	std::cout<<std::endl;
	std::vector<Scalar> stddevs_T, moys_T;
	for(int i = 0; i < nb_swatches; ++i)
	{
		Point topLeft = pointsTarget[i]-Point(swatch_width/2, swatch_height/2);
		Mat swatch = ImT_Lab.colRange(topLeft.x,topLeft.x+swatch_width).rowRange(topLeft.y,topLeft.y+swatch_height); 
		Scalar swatchMoy, swatchstddev;
		meanStdDev(swatch, swatchMoy, swatchstddev);
		moys_T.push_back(swatchMoy);
		stddevs_T.push_back(swatchstddev);		
		std::cout<<"MoysT :"<<swatchMoy<<std::endl;
		std::cout<<"stdT :"<<swatchstddev<<std::endl; 
	} 

	// computation of a target image for each swatch
	std::vector<Mat> Im_R_swatch;
	
	Mat ImR_Lab = Mat::zeros(imageTarget.rows, imageTarget.cols, CV_32FC3);

	for(int s = 0; s < nb_swatches; ++s)
	{
		MatIterator_<Vec3f> it_Rlab = ImR_Lab.begin<Vec3f>();
		MatIterator_<Vec3f> it_Tlab = ImT_Lab.begin<Vec3f>();

		Vec3f max(0,0,0);
		Vec3f min(numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max());

		Point targetPoint = pointsTarget[s] - Point(swatch_width / 2, swatch_height / 2);
		Point sourcePoint = pointsSource[s] - Point(swatch_width / 2, swatch_height / 2);

		Mat swatchTarget = ImT_Lab.colRange(targetPoint.x, targetPoint.x + swatch_width).rowRange(targetPoint.y, targetPoint.y + swatch_height);
		Mat swatchSource = ImS_Lab.colRange(sourcePoint.x, sourcePoint.x + swatch_width).rowRange(sourcePoint.y, sourcePoint.y + swatch_height);

		Mat MuTarget = Mat::zeros(2, 1, CV_32FC1);
		Mat MuSource = Mat::zeros(2, 1, CV_32FC1);
		Mat Transform = Mat::zeros(2, 2, CV_32FC1);

		computeTransformMatrix(swatchTarget, swatchSource, MuTarget, MuSource, Transform);


		for (; it_Rlab != ImR_Lab.end<Vec3f>(); ++it_Rlab, ++it_Tlab)
		{
				Scalar stddev_s = 0, moy_s = 0;
				Scalar stddev_t = 0, moy_t = 0;

				float alpha = 0.0f;

				Mat CTarget = Mat::zeros(2, 1, CV_32FC1);
				CTarget.at<float>(0, 0) = (*it_Tlab)[1];  // a
				CTarget.at<float>(1, 0) = (*it_Tlab)[2];  // b

				Mat MatTrans = (Transform * (CTarget - MuTarget)) + MuSource;
				(*it_Rlab)[0] = (*it_Tlab)[0]; // L
				(*it_Rlab)[1] = (*it_Tlab)[1] * alpha + (1 - alpha) * MatTrans.at<float>(0, 0); // a
				(*it_Rlab)[2] = (*it_Tlab)[2] * alpha + (1 - alpha) * MatTrans.at<float>(1, 0); // b 

				/*(*it_Rlab)[0] = (*it_Tlab)[0];
				(*it_Rlab)[1] = (stddevs_S[s][1] / (stddevs_T[s][1] == 0)?1 : stddevs_T[s][1]) * ((*it_Tlab)[1] - moys_T[s][1]) + moys_S[s][1];
				(*it_Rlab)[2] = (stddevs_S[s][2] / (stddevs_T[s][2] == 0)?1 : stddevs_T[s][2]) * ((*it_Tlab)[2] - moys_T[s][2]) + moys_S[s][2];
	*/
				


				// max &  computation
				max[0] = std::max((float)max[0], ((*it_Rlab)[0]));
				max[1] = std::max((float)max[1], ((*it_Rlab)[1]));
				max[2] = std::max((float)max[2], ((*it_Rlab)[2]));

				min[0] = std::min((float)min[0], ((*it_Rlab)[0]));
				min[1] = std::min((float)min[1], ((*it_Rlab)[1]));
				min[2] = std::min((float)min[2], ((*it_Rlab)[2]));
		}


		// normalize the swatch
		for (int i = 0; i < ImR_Lab.rows; i++)
		{
			for (int j = 0; j < ImR_Lab.cols; j++)
			{
				if (min[0] < 0 || max[0] > 100)
					ImR_Lab.at<Vec3f>(i, j)[0] = (ImR_Lab.at<Vec3f>(i, j)[0] - min[0]) / (max[0] - min[0]) * std::min(100.0f, max[0] - min[0]) + std::max(0.0f, min[0]);
				if (min[1] < -128 || max[1] > 127)
					ImR_Lab.at<Vec3f>(i, j)[1] = (ImR_Lab.at<Vec3f>(i, j)[1] - min[1]) / (max[1] - min[1]) * std::min(255.0f, max[1] - min[1]) + std::max(-128.0f, min[1]);
				if (min[2] < -128 || max[2] > 127)
					ImR_Lab.at<Vec3f>(i, j)[2] = (ImR_Lab.at<Vec3f>(i, j)[2] - min[2]) / (max[2] - min[2]) * std::min(255.0f, max[2] - min[2]) + std::max(-128.0f, min[2]);
			}
		}
		

		Mat imageResult;
		cvtColor(ImR_Lab, imageResult, CV_Lab2BGR);


		// Display of the swatch i
		/*Mat imageResultInt = Mat(imageResult.size(), CV_8UC3);
		imageResult.convertTo(imageResultInt, CV_8UC3);*/
		stringstream ss;
		
		ss<<"Color Result Swatch "<<s;
		//namedWindow(ss.str(), CV_WINDOW_AUTOSIZE );// Create a window for display.
		//imshow(ss.str(), imageResultInt ); 
		
		namedWindow(ss.str(), CV_WINDOW_AUTOSIZE);
		imshow(ss.str(), imageResult);
		//Ecriture
		stringstream ss2;
		ss2 << "results/swatches/swatch_" << s << ".png";

		normalize(imageResult, imageResult, 0, 255, NORM_MINMAX);
		imwrite(ss2.str(), imageResult);

		Im_R_swatch.push_back(ImR_Lab);
	}


	// Choix du swatch pour chaque pixel
	float sigma;
	do
	{
		std::cout << "Entrez un sigma (0 pour ne pas faire de ponderation, -1 pour sortir) :" << std::endl;
		std::cin >> sigma;

		if(sigma < 0)
			break;

		if(nb_swatches > 1)
		{		
			MatIterator_<Vec3f> it_Tlab = ImT_Lab.begin<Vec3f>();
			MatIterator_<Vec3f> it_Rlab = ImR_Lab.begin<Vec3f>();
			for (; it_Rlab != ImR_Lab.end<Vec3f>(); ++it_Rlab, ++it_Tlab)
			{
				Point pos = it_Rlab.pos();

				// Application de la pond�ration
				if (sigma > 0)
				{
					std::vector<float> distances;
					float pondSum = 0;

					for (int k = 0; k < Im_R_swatch.size(); ++k)
					{
						Vec3f moy = Vec3f(moys_S.at(k)[0], moys_S.at(k)[1], moys_S.at(k)[2]);
						Vec3f Lab = (*it_Tlab) - moy;
						float currentDist = sqrt(Lab[0] * Lab[0] + Lab[1] * Lab[1] + Lab[2] * Lab[2]);

						distances.push_back(currentDist);
						pondSum += currentDist;
					}

					float sum_w = 0;
					std::vector<float> w;

					for (int i = 0; i < Im_R_swatch.size(); ++i)
					{
						//float wi = cv::exp(-2 * (float)(distances[i] * distances[i]) / (sigma*sigma));
						float wi = cv::exp(-2 * (float)(distances[i] * distances[i]) / (sigma*sigma));
						w.push_back(wi);
						sum_w += wi;
					}

					int i = 0;
					for (std::vector<Mat>::iterator it_swatches = Im_R_swatch.begin(); it_swatches != Im_R_swatch.end(); ++it_swatches, ++i)
					{
						// non local mean
						(*it_Rlab) += (*it_swatches).at<Vec3f>(pos)*(w[i] / sum_w);
					}
				}				// Winner takes all
				else
				{
					float distSwatch = numeric_limits<float>::max();
					int k = 0;
					for (std::vector<Mat>::iterator it_swatches = Im_R_swatch.begin(); it_swatches != Im_R_swatch.end(); ++it_swatches, ++k)
					{
						Vec3f moy = Vec3f(moys_S.at(k)[0], moys_S.at(k)[1], moys_S.at(k)[2]);
						Vec3f Lab = *it_Tlab - moy;

						float currentDist = sqrt(Lab[0] * Lab[0] + Lab[1] * Lab[1] + Lab[2] * Lab[2]);

						if (currentDist < distSwatch)
						{
							distSwatch = currentDist;
							(*it_Rlab) = (*it_swatches).at<Vec3f>(pos);
						}
					}
				}
			}
		}
		else
			ImR_Lab = Im_R_swatch[0];

		Mat imageResult;
		cvtColor(ImR_Lab, imageResult, CV_Lab2BGR);
		//affichage
		namedWindow("Color Result", CV_WINDOW_AUTOSIZE);
		imshow("Color Result", imageResult);


		normalize(imageResult, imageResult, 0, 255, NORM_MINMAX);
		imwrite("results/swatches/swatch_result.png", imageResult);
		waitKey(0);
	}
	while(nb_swatches > 1 && sigma >= 0);

}



void histo(Mat & srcLab,Mat & colorizedLab, Mat & targetLab, int colorType){
    
	
	int histSize = 256;
	float range[2] ;
	if(colorType == 0)
	{
		range[0] =-127.0f;
		range[1] = 128.0f; //the upper boundary is exclusive
	}
	else
	{
		range[0] = 0.0f;
		range[1] = 256.0f ; //the upper boundary is exclusive
	}

	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;

	vector<Mat> clab_planes;
	vector<Mat> tlab_planes;
	vector<Mat> lab_planes;
	split( colorizedLab, clab_planes );
	split( targetLab, tlab_planes );
	split( srcLab, lab_planes );
   
	Mat l_hist, a_hist, b_hist;
	Mat cl_hist, ca_hist, cb_hist;
	Mat tl_hist, ta_hist, tb_hist;


	calcHist( &lab_planes[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &lab_planes[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &lab_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );

	calcHist( &clab_planes[0], 1, 0, Mat(), cl_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &clab_planes[1], 1, 0, Mat(), ca_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &clab_planes[2], 1, 0, Mat(), cb_hist, 1, &histSize, &histRange, uniform, accumulate );

	calcHist( &tlab_planes[0], 1, 0, Mat(), tl_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &tlab_planes[1], 1, 0, Mat(), ta_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &tlab_planes[2], 1, 0, Mat(), tb_hist, 1, &histSize, &histRange, uniform, accumulate );

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat lhistImage( hist_h, hist_w, CV_32FC3, Scalar( 0,0,0) );
	Mat ahistImage( hist_h, hist_w, CV_32FC3, Scalar( 0,0,0) );
	Mat bhistImage( hist_h, hist_w, CV_32FC3, Scalar( 0,0,0) );
	
	normalize(l_hist, l_hist, 0, lhistImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(a_hist, a_hist, 0, ahistImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(b_hist, b_hist, 0, bhistImage.rows, NORM_MINMAX, -1, Mat() );

	normalize(cl_hist, cl_hist, 0, lhistImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(ca_hist, ca_hist, 0, ahistImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(cb_hist, cb_hist, 0, bhistImage.rows, NORM_MINMAX, -1, Mat() );

	normalize(tl_hist, tl_hist, 0, lhistImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(ta_hist, ta_hist, 0, ahistImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(tb_hist, tb_hist, 0, bhistImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{

		line( ahistImage, Point( bin_w*(i-1), hist_h - cvRound(a_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(a_hist.at<float>(i)) ),
						 Scalar( 255, 0, 0), 2, 8, 0  );

		line( bhistImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
						 Scalar( 255, 0,0 ), 2, 8, 0  );

		line( ahistImage, Point( bin_w*(i-1), hist_h - cvRound(ca_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(ca_hist.at<float>(i)) ),
						 Scalar( 0, 0, 255), 2, 8, 0  );

		line( bhistImage, Point( bin_w*(i-1), hist_h - cvRound(cb_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(cb_hist.at<float>(i)) ),
						 Scalar( 0, 0, 255), 2, 8, 0  );

		line( ahistImage, Point( bin_w*(i-1), hist_h - cvRound(ta_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(ta_hist.at<float>(i)) ),
						 Scalar( 0, 255, 0), 2, 8, 0  );

		line( bhistImage, Point( bin_w*(i-1), hist_h - cvRound(tb_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(tb_hist.at<float>(i)) ),
						 Scalar( 0, 255, 0), 2, 8, 0  );

		if(colorType ==1)
		{
			line( lhistImage, Point( bin_w*(i-1), hist_h - cvRound(l_hist.at<float>(i-1)) ) ,
					Point( bin_w*(i), hist_h - cvRound(l_hist.at<float>(i)) ),
					Scalar( 255, 0, 0), 2, 8, 0  );

			line( lhistImage, Point( bin_w*(i-1), hist_h - cvRound(cl_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(cl_hist.at<float>(i)) ),
						 Scalar( 0, 0, 255), 2, 8, 0  );

			line( lhistImage, Point( bin_w*(i-1), hist_h - cvRound(tl_hist.at<float>(i-1)) ) ,
						 Point( bin_w*(i), hist_h - cvRound(tl_hist.at<float>(i)) ),
						 Scalar( 0, 255, 0), 2, 8, 0  );
		}
	}

	if(colorType == 1)
	{
		namedWindow("calcHist L", CV_WINDOW_AUTOSIZE );
		imshow("calcHist l", lhistImage );
	}

	namedWindow("calcHist a", CV_WINDOW_AUTOSIZE );
	imshow("calcHist a", ahistImage );
	namedWindow("calcHist b ", CV_WINDOW_AUTOSIZE );
	imshow("calcHist b", bhistImage );

}
















void mouseHandlerSource(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
		if(((std::vector<Point> *) param)->size() < nb_swatches)
		{
			((std::vector<Point> *) param)->push_back(Point(x, y));
			Point topLeft = Point(x, y) - Point(swatch_width/2, swatch_height/2);
			Point bottomRight = Point(x, y) + Point(swatch_width/2, swatch_height/2);
			rectangle(imSourceDisplayed, topLeft, bottomRight, colors[(((std::vector<Point> *) param)->size()-1)%colors.size()], 2);
			imshow( "Color image source", imSourceDisplayed ); 
		}
    }
}

void mouseHandlerTarget(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
		if(((std::vector<Point> *) param)->size() < nb_swatches)
		{
			((std::vector<Point> *) param)->push_back(Point(x, y));
			Point topLeft = Point(x, y) - Point(swatch_width/2, swatch_height/2);
			Point bottomRight = Point(x, y) + Point(swatch_width/2, swatch_height/2);
			rectangle(imTargetDisplayed, topLeft, bottomRight, colors[(((std::vector<Point> *) param)->size()-1)%colors.size()],2);
			imshow( "Color image target", imTargetDisplayed ); 
		}
    }
}