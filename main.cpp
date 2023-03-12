#include "akaze.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>


void cudaAkazeDemo2(int argc, char** argv);
void cudaFastAkazeDemo2(int argc, char** argv);
void cvAkazeDemo2(int argc, char** argv);
void drawKeypoints(akaze::AkazeData& a, cv::Mat& img, cv::Mat& dst, const double alpha = 255.0);
void drawMatches(akaze::AkazeData& a, akaze::AkazeData& b, cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, 
    const bool horizontal = true, const double alpha = 255.0);
void drawMatches(std::vector<cv::KeyPoint>& a, std::vector<cv::KeyPoint>& b, std::vector<cv::DMatch>& matches, 
	cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal = true);


int main(int argc, char** argv)
{
    cudaAkazeDemo2(argc, argv);
    cudaFastAkazeDemo2(argc, argv);
	cvAkazeDemo2(argc, argv);

    CHECK(cudaDeviceReset());
    return 0;
}


void drawKeypoints(akaze::AkazeData& a, cv::Mat& img, cv::Mat& dst, const double alpha)
{
	akaze::AkazePoint* data = a.h_data;
	cv::merge(std::vector<cv::Mat>{ img, img, img }, dst);
    dst.convertTo(dst, CV_8UC3, alpha);
	for (int i = 0; i < a.num_pts; i++)
	{
		akaze::AkazePoint& p = data[i];
		cv::Point center(cvRound(p.x), cvRound(p.y));
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		cv::circle(dst, center, MAX(1, MIN(5, p.size)), color);
	}
}


void drawMatches(akaze::AkazeData& a, akaze::AkazeData& b, cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal, const double alpha)
{
	int num_pts = a.num_pts;
	akaze::AkazePoint* akaze1 = a.h_data;
	akaze::AkazePoint* akaze2 = b.h_data;
	const int h1 = img1.rows;
	const int h2 = img2.rows;
	const int w1 = img1.cols;
	const int w2 = img2.cols;
	if (horizontal)
	{
		cv::Mat cat_img = cv::Mat::zeros(std::max<int>(h1, h2), w1 + w2, CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(w1, 0, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	else
	{
		cv::Mat cat_img = cv::Mat::zeros(h1 + h2, std::max<int>(w1, w2), CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(0, h1, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	dst.convertTo(dst, CV_8UC3, alpha);

	// Filter by distance
	for (int i = 0; i < num_pts; i++)
	{
		int k = akaze1[i].match;
		//int d = akaze1[i].distance;
		if (k != -1)	//  && d - min_dist < threshold
		{
			cv::Point p1(cvRound(akaze1[i].x), cvRound(akaze1[i].y));
			cv::Point p2(cvRound(akaze2[k].x), cvRound(akaze2[k].y));
			if (horizontal)
				p2.x += w1;
			else
				p2.y += h1;
			cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
			cv::line(dst, p1, p2, color);
		}
	}
}


void drawMatches(std::vector<cv::KeyPoint>& a, std::vector<cv::KeyPoint>& b, std::vector<cv::DMatch>& matches,
	cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal)
{
	const int h1 = img1.rows;
	const int h2 = img2.rows;
	const int w1 = img1.cols;
	const int w2 = img2.cols;
	if (horizontal)
	{
		cv::Mat cat_img = cv::Mat::zeros(std::max<int>(h1, h2), w1 + w2, CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(w1, 0, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	else
	{
		cv::Mat cat_img = cv::Mat::zeros(h1 + h2, std::max<int>(w1, w2), CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(0, h1, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	dst.convertTo(dst, CV_8UC3);

	for (size_t i = 0; i < matches.size(); i++)
	{
		cv::Point p1, p2;
		p1.x = cvRound(a[matches[i].queryIdx].pt.x);
		p1.y = cvRound(a[matches[i].queryIdx].pt.y);
		p2.x = cvRound(b[matches[i].trainIdx].pt.x);
		p2.y = cvRound(b[matches[i].trainIdx].pt.y);
		if (horizontal)
			p2.x += w1;
		else
			p2.y += h1;
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		cv::line(dst, p1, p2, color);
	}
}


void cudaAkazeDemo2(int argc, char** argv)
{
	std::cout << "===== Registration by CUDA-AKAZE =====" << std::endl;
	int devNum = 0, imgSet = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	if (argc > 2)
		imgSet = std::atoi(argv[2]);

	// Read images using OpenCV
	cv::Mat limg, rimg;
	if (imgSet)
	{
		limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/right.pgm", cv::IMREAD_GRAYSCALE);
	}
	else
	{
		limg = cv::imread("data/img1.png", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/img2.png", cv::IMREAD_GRAYSCALE);
	}
    limg.convertTo(limg, CV_32FC1, 1.0 / 255.0);
	rimg.convertTo(rimg, CV_32FC1, 1.0 / 255.0);
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	// Configuration
	int max_npts = 10000;
	int noctaves = 4;		// Max octave
	int max_scale = 4;		// The number of sublevels per octave layer
	float per = 0.7f;		// Percentile level for the contrast factor
	float kcontrast = 0.03f;// The contrast factor parameter
	float soffset = 1.6f;	// Base scale offset (sigma units)
	bool reordering = true;	// Flag for reordering time steps
	float derivative_factor = 1.5f;	// Factor for the multiscale derivatives
	float dthreshold = 0.001f;		// Detector response threshold to accept point
	int diffusivity = 1;			// Diffusivity type
	int descriptor_pattern_size = 10;	// Actual patch size is 2*pattern_size*point.scale

	// Initial Cuda images and download images to device
	std::cout << "Initializing data..." << std::endl;
	initDevice(devNum);

	GpuTimer timer(0);
	int3 whp1, whp2;
	whp1.x = limg.cols; whp1.y = limg.rows; whp1.z = iAlignUp(whp1.x, 128);
	whp2.x = rimg.cols; whp2.y = rimg.rows; whp2.z = iAlignUp(whp2.x, 128);
	size_t size1 = whp1.y * whp1.z * sizeof(float);
	size_t size2 = whp2.y * whp2.z * sizeof(float);
	float* img1 = NULL;
	float* img2 = NULL;
	size_t tmp_pitch = 0;
	CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(float) * whp1.x, whp1.y));
	CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(float) * whp2.x, whp2.y));
	const size_t dpitch1 = sizeof(float) * whp1.z;
	const size_t spitch1 = sizeof(float) * whp1.x;
	const size_t dpitch2 = sizeof(float) * whp2.z;
	const size_t spitch2 = sizeof(float) * whp2.x;
	CHECK(cudaMemcpy2D(img1, dpitch1, limg.data, spitch1, spitch1, whp1.y, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(img2, dpitch2, rimg.data, spitch2, spitch2, whp2.y, cudaMemcpyHostToDevice));
	float t0 = timer.read();

	/* Reserve memory space for a whole bunch of SURF features. */
	akaze::AkazeData akaze_data1, akaze_data2;
	akaze::initAkazeData(akaze_data1, max_npts, true, true);
	akaze::initAkazeData(akaze_data2, max_npts, true, true);

	std::unique_ptr<akaze::Akazer> detector(new akaze::Akazer);
	detector->init(whp1, noctaves, max_scale, per, kcontrast, soffset, reordering, derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);

	int nrepeats = 100;
	float t1 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		detector->detectAndCompute(img1, akaze_data1, whp1, true);
		detector->detectAndCompute(img2, akaze_data2, whp2, true);
	}

	float t2 = timer.read();

	akaze::cuMatch(akaze_data1, akaze_data2);

	float t3 = timer.read();
	std::cout << "Number of features1: " << akaze_data1.num_pts << std::endl
		<< "Number of features2: " << akaze_data2.num_pts << std::endl;
	std::cout << "Time for allocating image memory:  " << t0 << std::endl
		<< "Time for allocating point memory:  " << t1 - t0 << std::endl
		<< "Time of detection and computation: " << (t2 - t1) / nrepeats << std::endl
		<< "Time of matching AKAZE keypoints:   " << (t3 - t2) / nrepeats << std::endl;

	// Show
	cv::Mat show1, show2, show_matched;
	drawKeypoints(akaze_data1, limg, show1, 255.0);
	drawKeypoints(akaze_data2, rimg, show2, 255.0);
	drawMatches(akaze_data1, akaze_data2, limg, rimg, show_matched, false, 255.0);
	cv::imwrite("data/akaze_show1.jpg", show1);
	cv::imwrite("data/akaze_show2.jpg", show2);
	cv::imwrite("data/akaze_show_matched.jpg", show_matched);

	// Free Sift data from device
	akaze::freeAkazeData(akaze_data1);
	akaze::freeAkazeData(akaze_data2);
	CHECK(cudaFree(img1));
	CHECK(cudaFree(img2));
}


void cudaFastAkazeDemo2(int argc, char** argv)
{
	std::cout << "===== Registration by CUDA-AKAZE-FAST =====" << std::endl;
	int devNum = 0, imgSet = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	if (argc > 2)
		imgSet = std::atoi(argv[2]);

	// Read images using OpenCV
	cv::Mat limg, rimg;
	if (imgSet)
	{
		limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/right.pgm", cv::IMREAD_GRAYSCALE);
	}
	else
	{
		limg = cv::imread("data/img1.png", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/img2.png", cv::IMREAD_GRAYSCALE);
	}
    // limg.convertTo(limg, CV_32FC1, 1.0 / 255.0);
	// rimg.convertTo(rimg, CV_32FC1, 1.0 / 255.0);
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	// Configuration
	int max_npts = 10000;
	int noctaves = 4;		// Max octave
	int max_scale = 4;		// The number of sublevels per octave layer
	float per = 0.7f;		// Percentile level for the contrast factor
	float kcontrast = 0.03f;// The contrast factor parameter
	float soffset = 1.6f;	// Base scale offset (sigma units)
	bool reordering = true;	// Flag for reordering time steps
	float derivative_factor = 1.5f;	// Factor for the multiscale derivatives
	float dthreshold = 0.001f;		// Detector response threshold to accept point
	int diffusivity = 1;			// Diffusivity type
	int descriptor_pattern_size = 10;	// Actual patch size is 2*pattern_size*point.scale

	// Initial Cuda images and download images to device
	std::cout << "Initializing data..." << std::endl;
	initDevice(devNum);

	GpuTimer timer(0);
	int3 whp1, whp2;
	whp1.x = limg.cols; whp1.y = limg.rows; whp1.z = iAlignUp(whp1.x, 128);
	whp2.x = rimg.cols; whp2.y = rimg.rows; whp2.z = iAlignUp(whp2.x, 128);
	size_t size1 = whp1.y * whp1.z * sizeof(unsigned char);
	size_t size2 = whp2.y * whp2.z * sizeof(unsigned char);
	unsigned char* img1 = NULL;
	unsigned char* img2 = NULL;
	size_t tmp_pitch = 0;
	CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(unsigned char) * whp1.x, whp1.y));
	CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(unsigned char) * whp2.x, whp2.y));
	const size_t dpitch1 = sizeof(unsigned char) * whp1.z;
	const size_t spitch1 = sizeof(unsigned char) * whp1.x;
	const size_t dpitch2 = sizeof(unsigned char) * whp2.z;
	const size_t spitch2 = sizeof(unsigned char) * whp2.x;
	CHECK(cudaMemcpy2D(img1, dpitch1, limg.data, spitch1, spitch1, whp1.y, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(img2, dpitch2, rimg.data, spitch2, spitch2, whp2.y, cudaMemcpyHostToDevice));
	float t0 = timer.read();

	/* Reserve memory space for a whole bunch of SURF features. */
	akaze::AkazeData akaze_data1, akaze_data2;
	akaze::initAkazeData(akaze_data1, max_npts, true, true);
	akaze::initAkazeData(akaze_data2, max_npts, true, true);

	std::unique_ptr<akaze::Akazer> detector(new akaze::Akazer);
	detector->init(whp1, noctaves, max_scale, per, kcontrast, soffset, reordering, derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);

	int nrepeats = 100;
	float t1 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		detector->fastDetectAndCompute(img1, akaze_data1, whp1, true);
		detector->fastDetectAndCompute(img2, akaze_data2, whp2, true);
	}

	float t2 = timer.read();

	akaze::cuMatch(akaze_data1, akaze_data2);

	float t3 = timer.read();
	std::cout << "Number of features1: " << akaze_data1.num_pts << std::endl
		<< "Number of features2: " << akaze_data2.num_pts << std::endl;
	std::cout << "Time for allocating image memory:  " << t0 << std::endl
		<< "Time for allocating point memory:  " << t1 - t0 << std::endl
		<< "Time of detection and computation: " << (t2 - t1) / nrepeats << std::endl
		<< "Time of matching AKAZE keypoints:   " << (t3 - t2) / nrepeats << std::endl;

	// Show
	cv::Mat show1, show2, show_matched;
	drawKeypoints(akaze_data1, limg, show1, 1);
	drawKeypoints(akaze_data2, rimg, show2, 1);
	drawMatches(akaze_data1, akaze_data2, limg, rimg, show_matched, false, 1);
	cv::imwrite("data/fastakaze_show1.jpg", show1);
	cv::imwrite("data/fastakaze_show2.jpg", show2);
	cv::imwrite("data/fastakaze_show_matched.jpg", show_matched);

	// Free Sift data from device
	akaze::freeAkazeData(akaze_data1);
	akaze::freeAkazeData(akaze_data2);
	CHECK(cudaFree(img1));
	CHECK(cudaFree(img2));
}


void cvAkazeDemo2(int argc, char** argv)
{
	std::cout << "===== Registration by OpenCV =====" << std::endl;
	// Did not very good! Too many error matches
	int devNum = 0, imgSet = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	if (argc > 2)
		imgSet = std::atoi(argv[2]);

	// Read images using OpenCV
	cv::Mat limg, rimg;
	if (imgSet)
	{
		limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/right.pgm", cv::IMREAD_GRAYSCALE);
	}
	else
	{
		limg = cv::imread("data/img1.png", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/img2.png", cv::IMREAD_GRAYSCALE);
	}
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	int nrepeats = 100;
	cv::Mat desc_r, desc_t;
	std::vector<cv::KeyPoint> kpt_r, kpt_t;
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	std::vector<cv::DMatch> imatches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

	GpuTimer timer(0);
	for (int i = 0; i < nrepeats; i++)
	{
		detector->detectAndCompute(limg, cv::Mat(), kpt_r, desc_r);
		detector->detectAndCompute(rimg, cv::Mat(), kpt_t, desc_t);
	}

	float t0 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		matcher->match(desc_r, desc_t, imatches);
	}

	float t1 = timer.read();
	cv::Mat show_matched;
	drawMatches(kpt_r, kpt_t, imatches, limg, rimg, show_matched, false);
	cv::imwrite("data/cvshow_matched.jpg", show_matched);

	std::cout << "Number of original features: " << kpt_t.size() << " " << kpt_r.size() << std::endl;
	std::cout << "Number of matching features: " << imatches.size() << std::endl;
	std::cout << "Time of detection and computation: " << t0 / nrepeats << std::endl
		<< "Time of matching akaze keypoints:   " << (t1 - t0) / nrepeats << std::endl;
}