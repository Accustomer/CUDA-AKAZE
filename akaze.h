#pragma once
#include "akaze_structures.h"
#include "cuda_utils.h"



namespace akaze
{

	void initAkazeData(AkazeData& data, const int max_pts, const bool host, const bool dev);

	void freeAkazeData(AkazeData& data);

	void cuMatch(AkazeData& result1, AkazeData& result2);



	class Akazer
	{
	public:
		Akazer();
		~Akazer();

		/* Initialization */
		void init(int3 whp0, int _noctaves, int _max_scale, float _per, float _kcontrast, float _soffset, bool _reordering, 
			float _derivative_factor, float _dthreshold, int _diffusivity, int _descriptor_pattern_size);

		/* Detect keypoints and compute features */
		void detectAndCompute(float* image, AkazeData& result, int3 whp0, const bool desc = true);
		void fastDetectAndCompute(unsigned char* image, AkazeData& result, int3 whp0, const bool desc = true);

	private:
		// Size of image
		int3 whp{0, 0, 0};
		// Max octave
		int noctaves = 4;
		// The number of sublevels per octave layer
		int max_scale = 4;
		// Percentile level for the contrast factor
		float per = 0.7f;
		// The contrast factor parameter
		float kcontrast = 0.03f;
		// Base scale offset (sigma units)
		float soffset = 1.6f;
		// Flag for reordering time steps
		bool reordering = true;
		// Factor for the multiscale derivatives
		float derivative_factor = 1.5f;
		// Detector response threshold to accept point
		float dthreshold = 0.001f;
		// Diffusivity type
		DiffusivityType diffusivity = PM_G2;
		// Actual patch size is 2*pattern_size*point.scale
		int descriptor_pattern_size = 10;
		// Memory for images in octave
		float* omem = NULL;
		size_t total_osize = 0;


		/* Allocate memory */
		void allocMemory(void** addr, int3& whp0, int3* owhps, int* osizes, int* offsets, const bool reused);

		/* Create nonlinear scale space */
		void detect(AkazeData& result, float* tmem, float* image, int3* owhps, int* osizes, int* offsets);
		void fastDetect(AkazeData& result, void* tmem, unsigned char* image, int3* owhps, int* osizes, int* offsets);

	};


}
