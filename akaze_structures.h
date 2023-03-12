#pragma once


namespace akaze
{

/*
0 - SURF_UPRIGHT
1 - SURF
2 - MSURF_UPRIGHT
3 - MSURF
4 - MLDB_UPRIGHT
5 - MLDB
*/
#define FEATURE_TYPE 5


	/* Structure of AKAZE point */
	struct AkazePoint
	{
		float x;
		float y;
		int octave;
		float response;
		float size;
		float angle;

#if (FEATURE_TYPE == 5)
#define FLEN 61
		unsigned char features[FLEN];
#else
#define FLEN 64
		float features[FLEN];
#endif // (FEATURE_TYPE == 5)

		int match;
		int distance;
		float match_x;
		float match_y;
	};


	/* Structure of AKAZE matching data */
	struct AkazeData
	{
		int num_pts;		// Number of available AKAZE points
		int max_pts;		// Number of allocated AKAZE points
		AkazePoint* h_data;	// Host (CPU) data
		AkazePoint* d_data;	// Device (GPU) data
	};


	enum DiffusivityType
	{
		PM_G1 = 0,
		PM_G2 = 1,
		WEICKERT = 2,
		CHARBONNIER = 3
	};

}
