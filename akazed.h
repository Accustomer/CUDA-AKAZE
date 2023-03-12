#pragma once
#include "akaze_structures.h"
#include "cuda_utils.h"


/* Set the maximum number of keypoints. */
void setMaxNumPoints(const int num);

/* Get the address of point counter */
void getPointCounter(void** addr);

/* Get the address of max contrast value */
void getMaxContrastAddr(void** addr);

/* Set the data of histogram */
void setHistogram(const int* h_hist);

/* Copy paramters for extrema points extraction */
void setExtremaParam(const float* param, const int n);

/* Set parameters of octave */
void setOparam(const int* oparams, const int n);

/* Initialize compare indices */
void setCompareIndices();



namespace akaze
{
	/* Set the kernel of low-pass */
	void setLowPassKernel(const float* kernel, const int ksz);
	
	/* Gaussian convolution */
	void hConv2d(float* src, float* dst, int width, int height, int pitch);
	void hSepConv2d(float* src, float* dst, int width, int height, int pitch);
	void hLowPass(float* src, float* dst, int width, int height, int pitch, float var, int ksz);

	/* 
	Downsample with gaussian scale 
	@Param:
		w/h/p 1 - Size of destination image
		p0 - Size of source image
	*/
	void hDownWithSmooth(float* src, float* dst, float* smooth, int3 swhp, int3 dwhp);

	/* Apply scharr filter to compute gradient */
	void hScharrContrast(float* src, float* grad, float& kcontrast, float per, int width, int height, int pitch);

	/* Compute diffusivity flow */
	void hFlow(float* src, float* flow, DiffusivityType type, float kcontrast, int width, int height, int pitch);

	/* Apply NLD step */
	void hNldStep(float* img, float* flow, float* temp, float step_size, int width, int height, int pitch);

	/* Hessian determinant */
	void hHessianDeterminant(float* src, float* dx, float* dy, int step, int width, int height, int pitch);

	/* Compute extrema map */
	void hCalcExtremaMap(float* dets, float* response_map, float* size_map, int* layer_map, float* params,
		int octave, int max_scale, float threshold, int width, int height, int pitch, int opitch);

	/* Apply nms and compute keypoints */
	void hNms(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int width, int height, int pitch);
	void hNmsR(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int neigh, int width, int height, int pitch);

	/* Refine points */
	void hRefine(AkazeData& result, float* tmem, int noctaves, int max_scale);

	/* Compute orientation */
	void hCalcOrient(AkazeData& result, float* tmem, int noctaves, int max_scale);

	/* Extract descriptors */
	void hDescribe(AkazeData& result, float* tmem, int noctaves, int max_scale, int patsize);

	/* Match descriptors */
	void hMatch(AkazeData& result1, AkazeData& result2);

}




namespace fastakaze
{

	/* Gaussian convolution */
	void hConv2dR2(unsigned char* src, int* dst, int width, int height, int pitch, float var);
	void hConv2dR2(int* src, int* dst, int width, int height, int pitch, float var);
	void hConv2dR2(unsigned char* src, int* dst, int* temp, int width, int height, int pitch, float var);
	void hConv2dR2(int* src, int* dst, int* temp, int width, int height, int pitch, float var);
	void hLowPass(unsigned char* src, int* dst, int width, int height, int pitch, float var, int ksz);
	void hLowPass(unsigned char* src, int* dst, int* temp, int width, int height, int pitch, float var, int ksz);

	/* Downsample with gaussian scale */
	void hDownWithSmooth(int* src, int* dst, int* smooth, int3 swhp, int3 dwhp);

	/* Apply scharr filter to compute gradient */
	void hScharrContrast(int* src, int* grad, int& kcontrast, float per, int width, int height, int pitch);

	/* Hessian determinant */
	void hHessianDeterminant(int* src, int* dx, int* dy, int step, int width, int height, int pitch);

	/* Compute diffusivity flow */
	void hFlow(int* src, int* flow, akaze::DiffusivityType type, int kcontrast, int width, int height, int pitch);

	/* Apply NLD step */
	void hNldStep(int* img, int* flow, int* temp, float step_size, int width, int height, int pitch);

	/* Compute extrema map */
	void hCalcExtremaMap(int* dets, int* response_map, float* size_map, int* layer_map, float* params,
		int octave, int max_scale, int threshold, int width, int height, int pitch, int opitch);

	/* Apply nms and compute keypoints */
	void hNmsR(akaze::AkazePoint* points, int* response_map, float* size_map, int* layer_map, int psz, int neigh, int width, int height, int pitch);

	/* Refine points */
	void hRefine(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale);

	/* Compute orientation */
	void hCalcOrient(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale);

	/* Extract descriptors */
	void hDescribe(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale, int patsize);

}