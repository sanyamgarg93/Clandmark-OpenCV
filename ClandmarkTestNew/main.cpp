//Including OpenCV Header files
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <direct.h>
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>

//Including Flandmark and Clandmark header files
#include "Flandmark.h"
#include "CFeaturePool.h"
#include "CSparseLBPFeatures.h"
#include "helpers.h"

using namespace std;
using namespace cv;
using namespace clandmark;

string face_cascade_name;
CascadeClassifier face_cascade;

//Converting Mat format to CImg
cimg_library::CImg<unsigned char> * cvImgToCImg(cv::Mat &cvImg)
{
	cimg_library::CImg<unsigned char> * result = new cimg_library::CImg<unsigned char>(cvImg.cols, cvImg.rows);

	for (int x = 0; x < cvImg.cols; ++x)
	for (int y = 0; y < cvImg.rows; ++y)
		(*result)(x, y) = cvImg.at<uchar>(y, x);

	return result;
}

//Running the face detection and landmark detection on every frame of the input feed
void detectAndDisplay(Mat &frame, Flandmark *flandmark, CFeaturePool *featurePool)
{	
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	equalizeHist(frame_gray, frame_gray);

	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, Size(100, 100));

	for (uint32_t i = 0; i < faces.size(); i++)
	{
		// Get detected face bounding box
		int bbox[8];		
		bbox[0] = faces[i].x;
		bbox[1] = faces[i].y;
		bbox[2] = faces[i].x + faces[i].width;
		bbox[3] = faces[i].y;
		bbox[4] = faces[i].x + faces[i].width;
		bbox[5] = faces[i].y + faces[i].height;
		bbox[6] = faces[i].x;
		bbox[7] = faces[i].y + faces[i].height;

		// Detect facial landmarks
		cimg_library::CImg<unsigned char>* frm_gray = cvImgToCImg(frame_gray);
		flandmark->detect_optimized(frm_gray, bbox);

		//flandmark->detect_from_nf(frm_gray, bbox);
		//flandmark->detect( frm_gray, bbox );

		delete frm_gray;

		// Get detected landmarks
		fl_double_t *landmarks = flandmark->getLandmarks();

		// Draw bounding box and detected landmarks
		for (int i = 0; i < 2 * flandmark->getLandmarksCount(); i += 2)
		{
			circle(frame, Point(int(landmarks[i]), int(landmarks[i + 1])), 2, Scalar(0, 0, 255), -1);
		}

		// Textual output
		//printTimingStats(flandmark->timings);
		//printLandmarks(landmarks, flandmark->getLandmarksCount());
		//printLandmarks(flandmark->getLandmarksNF(), flandmark->getLandmarksCount());
	}
}

int main()
{
	//Extract the ProjectDir path since the cascades are stored in $(ProjectDir)\DetectionCascades
	char cProjectDirPath[FILENAME_MAX];
	if (!_getcwd(cProjectDirPath, sizeof(cProjectDirPath)))
		return errno;
	
	cProjectDirPath[sizeof(cProjectDirPath)-1] = '\0'; 
	string str(cProjectDirPath);
	cout << "DetectionCascades path:" << str + "\\DetectionCascades" << endl;

	string CPDMCascadeStr = str + "\\DetectionCascades\\CDPM.xml";
	const char * cCPDMCascadeStr = CPDMCascadeStr.c_str();

	string faceCascadeStr = str + "\\DetectionCascades\\lbpcascade_frontalface.xml";
	const char * cFaceCascadeStr = faceCascadeStr.c_str();

	double tim = (double)getTickCount();

	//Initialize and load clandmark library
	Flandmark *flandmark = Flandmark::getInstanceOf(cCPDMCascadeStr);
	if (!flandmark)
	{
		cerr << "Could not load the Clandmark Lib" << endl; cin.get();
		return -1;
	}

	CFeaturePool *featurePool = new CFeaturePool(flandmark->getBaseWindowSize()[0], flandmark->getBaseWindowSize()[1]);
	featurePool->addFeaturesToPool(
		new CSparseLBPFeatures(
		featurePool->getWidth(),
		featurePool->getHeight(),
		featurePool->getPyramidLevels(),
		featurePool->getCumulativeWidths()
		));

	flandmark->setNFfeaturesPool(featurePool);

	tim = ((double)getTickCount() - tim) / getTickFrequency() * 1000;
	cout << "Flandmark model loaded in " << tim << " ms" << endl;

	//Load Face Detection Cascade
	if (!face_cascade.load(cFaceCascadeStr))
	{
		printf("Could not load face detection cascade\n"); cin.get();
		return -1;
	};

	//VideoCapture capture(0); //Use Webcam
	VideoCapture capture("talking_face.avi");	//Talking Face Database
	Mat image;

	while (true)
	{
		if (!capture.read(image))
		{
			cout << "Cannot read frames from webcam" << endl;
			cin.get();
		}

		if (!image.empty())
		{
			pyrDown(image, image, Size());
			detectAndDisplay(image, flandmark, featurePool);
		}
		else
		{
			cout << "Wrong input." << endl;
		}

		namedWindow("Output", CV_WINDOW_AUTOSIZE);
		imshow("Output", image);
		if (waitKey(10) == 27)
			break;
	}

	//Destruct
	delete featurePool;
	delete flandmark;
	return 0;
}