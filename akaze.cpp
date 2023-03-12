#include "akaze.h"
#include "akazed.h"
#include "fed.h"
#include <memory>
#include <cmath>

//#define DEBUG_SHOW

#ifdef DEBUG_SHOW
#include <opencv2/core.hpp>
#endif // DEBUG_SHOW


#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif



namespace akaze
{
	void initAkazeData(AkazeData& data, const int max_pts, const bool host, const bool dev)
	{
		data.num_pts = 0;
		data.max_pts = max_pts;
		const size_t size = sizeof(AkazePoint) * max_pts;
		data.h_data = host ? (AkazePoint*)malloc(size) : NULL;
		data.d_data = NULL;
		if (dev)
		{
			CHECK(cudaMalloc((void**)&data.d_data, size));
		}
	}


	void freeAkazeData(AkazeData& data)
	{
		if (data.d_data != NULL)
		{
			CHECK(cudaFree(data.d_data));
		}
		if (data.h_data != NULL)
		{
			free(data.h_data);
		}
		data.num_pts = 0;
		data.max_pts = 0;
	}


	void cuMatch(AkazeData& result1, AkazeData& result2)
	{
		hMatch(result1, result2);
		if (result1.h_data)
		{
			int* h_ptr = &result1.h_data[0].match;
			int* d_ptr = &result1.d_data[0].match;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), 4 * sizeof(float), result1.num_pts, cudaMemcpyDeviceToHost));
		}
	}




	Akazer::Akazer()
	{
	}


	Akazer::~Akazer()
	{
		CHECK(cudaFree(omem));
	}


	void Akazer::init(int3 whp0, int _noctaves, int _max_scale, float _per, float _kcontrast, float _soffset, bool _reordering,
		float _derivative_factor, float _dthreshold, int _diffusivity, int _descriptor_pattern_size)
	{
		whp.x = whp0.x;
		whp.y = whp0.y;
		whp.z = whp0.z;
		noctaves = _noctaves;
		max_scale = _max_scale;
		per = _per;
		kcontrast = _kcontrast;
		soffset = _soffset;
		reordering = _reordering;
		derivative_factor = _derivative_factor;
		dthreshold = _dthreshold;
		diffusivity = DiffusivityType(_diffusivity);
		descriptor_pattern_size = _descriptor_pattern_size;

		setCompareIndices();
	}


	void Akazer::detectAndCompute(float* image, AkazeData& result, int3 whp0, const bool desc)
	{
		// Allocate memory 
		std::unique_ptr<int> oparams(new int[noctaves * 5 + 1]);
		int* osizes = oparams.get();
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);		
		float* tmem = NULL;
		const bool reused = whp0.x == whp.x && whp0.y == whp.y;
		if (reused)
		{
			this->allocMemory((void**)&omem, whp0, owhps, osizes, offsets, reused);
			tmem = omem;
		}
		else
		{
			this->allocMemory((void**)&tmem, whp0, owhps, osizes, offsets, reused);
		}

		// Detect keypoints
		this->detect(result, tmem, image, owhps, osizes, offsets);

		// Compute descriptors
		if (desc)
		{
			// Compute orientations
			hCalcOrient(result, tmem, noctaves, max_scale);

			// Compute descriptors
			hDescribe(result, tmem, noctaves, max_scale, descriptor_pattern_size);
		}

		// Copy point data to host
		if (result.h_data != NULL)
		{
			float* h_ptr = &result.h_data[0].x;
			float* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), (desc ? FLEN * sizeof(unsigned char) : 0) + 6 * sizeof(float), result.num_pts, cudaMemcpyDeviceToHost));
		}

		// Post-processing
		if (reused)
		{
			CHECK(cudaMemset(omem, 0, total_osize));
		}
		else
		{
			CHECK(cudaFree(tmem));
		}
	}


	void Akazer::fastDetectAndCompute(unsigned char* image, AkazeData& result, int3 whp0, const bool desc)
	{
		// Allocate memory 
		std::unique_ptr<int> oparams(new int[noctaves * 5 + 1]);
		int* osizes = oparams.get();
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		void* tmem = NULL;
		const bool reused = whp0.x == whp.x && whp0.y == whp.y;
		if (reused)
		{
			this->allocMemory((void**)&omem, whp0, owhps, osizes, offsets, reused);
			tmem = omem;
		}
		else
		{
			this->allocMemory((void**)&tmem, whp0, owhps, osizes, offsets, reused);
		}

		// Detect keypoints
		this->fastDetect(result, tmem, image, owhps, osizes, offsets);

		// Compute descriptors
		if (desc)
		{
			// Compute orientations
			fastakaze::hCalcOrient(result, tmem, noctaves, max_scale);
			// Compute descriptors
			fastakaze::hDescribe(result, tmem, noctaves, max_scale, descriptor_pattern_size);
		}

		// Copy point data to host
		if (result.h_data != NULL)
		{
			float* h_ptr = &result.h_data[0].x;
			float* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), (desc ? FLEN * sizeof(unsigned char) : 0) + 6 * sizeof(float), result.num_pts, cudaMemcpyDeviceToHost));
		}

		// Post-processing
		if (reused)
		{
			CHECK(cudaMemset(omem, 0, total_osize));
		}
		else
		{
			CHECK(cudaFree(tmem));
		}
	}


	void Akazer::allocMemory(void** addr, int3& whp0, int3* owhps, int* osizes, int* offsets, const bool reused)
	{
		// Compute sizes
		owhps[0] = whp0;
		osizes[0] = whp0.y * whp0.z;
		offsets[0] = 3 * osizes[0];	// Record response, scale and octave map for NMS
		offsets[1] = offsets[0] + osizes[0] * max_scale * 4;	// 4: Limg, Lsmooth, Lx, Ly		
		for (int i = 0, j = 1, k = 2; j < noctaves; i++, j++, k++)
		{
			owhps[j].x = (owhps[i].x >> 1);
			owhps[j].y = (owhps[i].y >> 1);
			if (owhps[j].x < 80 || owhps[j].y < 80) 
			{
				noctaves = j;
				break;
			}
			owhps[j].z = iAlignUp(owhps[j].x, 128);
			osizes[j] = owhps[j].y * owhps[j].z;
			offsets[k] = offsets[j] + osizes[j] * max_scale * 4;
		}

		// Allocate memory for images in octave
		if ((reused && !omem) || !reused)
		{
			CHECK(cudaMalloc(addr, offsets[noctaves] * sizeof(float)));
		}

		if (reused)
		{
			total_osize = offsets[noctaves] * sizeof(float);
		}

		//return offsets[noctaves];
	}


	void Akazer::detect(AkazeData& result, float* tmem, float* image, int3* owhps, int* osizes, int* offsets)
	{
		// Get address of point counter
		unsigned int* d_point_counter_addr;
		getPointCounter((void**)&d_point_counter_addr);
		CHECK(cudaMemset(d_point_counter_addr, 0, sizeof(unsigned int)));
		setMaxNumPoints(result.max_pts);

		int w, h, p, msz, ms_msz, mstep;
		float* response_map = tmem;
		float* size_map = tmem + osizes[0];
		int* layer_map = (int*)(size_map + osizes[0]);

		size_t nbytes = osizes[0] * sizeof(float);
		float minv = 1e-6f;
		int* iminv = (int*)&minv;
		CHECK(cudaMemset(layer_map, -1, nbytes));
		CHECK(cudaMemset(response_map, *iminv, nbytes));
		CHECK(cudaMemset(size_map, *iminv, nbytes));
		
		float* oldnld = NULL;
		float* nldimg = NULL;
		float* smooth = NULL;
		float* flow = NULL;
		float* temp = NULL;
		float* dx = NULL;
		float* dy = NULL;

		float tmax = 0.25f;
		float esigma = soffset;
		float last_etime = 0.5 * soffset * soffset;
		float curr_etime = 0;
		float ttime = 0;
		int naux = 0;
		int oratio = 1;
		int sigma_size = 0;

		float smax = 1.0f;
		if (FEATURE_TYPE == 0 || FEATURE_TYPE == 1 || FEATURE_TYPE == 4 || FEATURE_TYPE == 5)
		{
			smax = 10.0 * sqrtf(2.0f);
		}
		else if (FEATURE_TYPE == 2 || FEATURE_TYPE == 3)
		{
			smax = 12.0 * sqrtf(2.0f);
		}
		std::unique_ptr<float> exptr(new float[max_scale * 2]);
		float* borders = exptr.get();
		float* sizes = borders + max_scale;
		float psz = 10000;
		int neigh = 0;
		//float* threshs = borders + max_scale;

#ifdef DEBUG_SHOW
		cv::Mat nldshow, detshow, dxshow, dyshow;
		cv::Mat response_show(owhps[0].y, owhps[0].x, CV_32FC1);
		cv::Mat size_show(owhps[0].y, owhps[0].x, CV_32FC1); 
		cv::Mat layer_show(owhps[0].y, owhps[0].x, CV_32SC1);;
#endif // DEBUG_SHOW	

		for (int i = 0; i < noctaves; i++)
		{
			w = owhps[i].x;
			h = owhps[i].y;
			p = owhps[i].z;
			msz = osizes[i];
			ms_msz = msz * max_scale;
			
#ifdef DEBUG_SHOW
			nldshow.create(h, w, CV_32FC1);
			detshow.create(h, w, CV_32FC1);
			dxshow.create(h, w, CV_32FC1);
			dyshow.create(h, w, CV_32FC1);
#endif // DEBUG_SHOW

			nldimg = tmem + offsets[i];
			smooth = nldimg + ms_msz;
			flow = smooth + ms_msz;
			temp = flow + ms_msz;
			dx = flow;
			dy = temp;

			// Create nonlinear space for current octave layer
			for (int j = 0; j < max_scale; j++)
			{
				if (j == 0 && i == 0)
				{
					float var = soffset * soffset;
					int ksz = 2 * ceilf((soffset - 0.8f) / 0.3f) + 3;
					hLowPass(image, smooth, w, h, p, 1.f, 5);
					hScharrContrast(smooth, temp, kcontrast, per, w, h, p);
					hLowPass(image, nldimg, w, h, p, var, ksz);
					CHECK(cudaMemcpy(smooth, nldimg, msz * sizeof(float), cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(float) * w, nldimg, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(float) * w, smooth, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(float) * w, temp, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW

					// Compute Hessian Determinant
					sizes[j] = esigma * derivative_factor;
					sigma_size = (int)(esigma * derivative_factor + 0.5f);
					borders[j] = smax * sigma_size;
					hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(float) * w, nldimg, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(float) * w, smooth, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(float) * w, dx, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dyshow.data, sizeof(float) * w, dy, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW

					continue;
				}

				std::vector<float> tau;
				esigma = soffset * powf(2, (float)j / max_scale + i);
				curr_etime = 0.5f * esigma * esigma;
				ttime = curr_etime - last_etime;
				naux = fed_tau_by_process_time(ttime, 1, tmax, reordering, tau);
				sizes[j] = esigma * derivative_factor / oratio;
				sigma_size = (int)(sizes[j] + 0.5f);
				borders[j] = smax * sigma_size;

				//for (int k = 0; k < naux; k++)
				//{
				//	printf("%f ", tau[k]);
				//}
				//printf("\n");

				if (j == 0)
				{
					kcontrast *= 0.75f;
					oldnld = nldimg - mstep;
					hDownWithSmooth(oldnld, nldimg, smooth, owhps[i - 1], owhps[i]);
					hFlow(smooth, flow, diffusivity, kcontrast, w, h, p);

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(float) * w, nldimg, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(float) * w, smooth, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(float) * w, dx, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW
					for (int k = 0; k < naux; k++)
					{
						hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(float), cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SHOW
						CHECK(cudaMemcpy2D(nldshow.data, sizeof(float) * w, nldimg, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW
					}
				}
				else
				{
					oldnld = nldimg;
					nldimg += msz;
					smooth += msz;
					flow += msz;
					temp += msz;
					dx = flow;
					dy = temp;

					hLowPass(oldnld, smooth, w, h, p, 1.f, 5);
					hFlow(smooth, flow, diffusivity, kcontrast, w, h, p);
					hNldStep(oldnld, flow, nldimg, tau[0], w, h, p);		

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(float) * w, nldimg, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(float) * w, smooth, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(float) * w, flow, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW
					for (int k = 1; k < naux; k++)
					{
						hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(float), cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SHOW
						CHECK(cudaMemcpy2D(nldshow.data, sizeof(float) * w, nldimg, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW						
					}
				}

				hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);

#ifdef DEBUG_SHOW
				CHECK(cudaMemcpy2D(detshow.data, sizeof(float) * w, smooth, sizeof(float) * p, sizeof(float) * w, h, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW
				last_etime = curr_etime;
			}

			// Detect keypoints for current octave layer
			float* dets = tmem + offsets[i] + ms_msz;	// (smooth - ms_msz)
			hCalcExtremaMap(dets, response_map, size_map, layer_map, borders, i, max_scale, dthreshold, w, h, p, owhps[0].z);
			psz = MIN(psz, borders[0] * oratio);
			neigh = MAX(neigh, sigma_size);

			mstep = ms_msz * 4;
			oratio *= 2;
		}

#ifdef DEBUG_SHOW
		CHECK(cudaMemcpy2D(response_show.data, sizeof(float)* owhps[0].x, response_map, sizeof(float)* owhps[0].z, sizeof(float)* owhps[0].x, owhps[0].y, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy2D(size_show.data, sizeof(float)* owhps[0].x, size_map, sizeof(float)* owhps[0].z, sizeof(float)* owhps[0].x, owhps[0].y, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy2D(layer_show.data, sizeof(float)* owhps[0].x, layer_map, sizeof(float)* owhps[0].z, sizeof(float)* owhps[0].x, owhps[0].y, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW

		// NMS
		//hNms(result.d_data, response_map, size_map, layer_map, (int)psz, owhps[0].x, owhps[0].y, owhps[0].z);
		hNmsR(result.d_data, response_map, size_map, layer_map, (int)psz, neigh, owhps[0].x, owhps[0].y, owhps[0].z);
		CHECK(cudaMemcpy(&result.num_pts, d_point_counter_addr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		result.num_pts = MIN(result.num_pts, result.max_pts);

		// Refine points
		setOparam(osizes, 5 * noctaves + 1);
		hRefine(result, tmem, noctaves, max_scale);

//		float* h_ptr = &result.h_data[0].x;
//		float* d_ptr = &result.d_data[0].x;
//		CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), 6 * sizeof(float), result.num_pts, cudaMemcpyDeviceToHost));
//		for (int pi = 0; pi < result.num_pts; pi++)
//		{
//			AkazePoint* pt = result.h_data + pi;
//			int o = pt->octave / max_scale;
//			int s = pt->octave % max_scale;
//			int p = owhps[o].z;
//			float* det = tmem + offsets[o] + (max_scale + s) * osizes[o];
//#ifdef DEBUG_SHOW
//			int w = owhps[o].x;
//			int h = owhps[o].y;
//			detshow = cv::Mat(h, p, CV_32FC1);
//			CHECK(cudaMemcpy(detshow.data, det, sizeof(float) * p * h, cudaMemcpyDeviceToHost));
//			//CHECK(cudaMemcpy2D(detshow.data, sizeof(float)* w, det, sizeof(float)* p, sizeof(float)* w, h, cudaMemcpyDeviceToHost));
//			det = (float*)detshow.data;
//#endif // DEBUG_SHOW
//			int y = (int)pt->y >> o;
//			int x = (int)pt->x >> o;
//			int idx = y * p + x;
//			//int idx = (int)pt->y * p + (int)pt->x;
//			float v2 = det[idx] + det[idx];
//			float dx = 0.5f * (det[idx + 1] - det[idx - 1]);
//			float dy = 0.5f * (det[idx + p] - det[idx - p]);
//			float dxx = det[idx + 1] + det[idx - 1] - v2;
//			float dyy = det[idx + p] + det[idx - p] - v2;
//			float dxy = 0.25f * (det[idx + p + 1] + det[idx - p - 1] - det[idx - p + 1] - det[idx + p - 1]);
//			float dd = dxx * dyy - dxy * dxy;
//			float idd = dd != 0.f ? 1.f / dd : 0.f;
//			float dst0 = idd * (dxy * dy - dyy * dx);
//			float dst1 = idd * (dxy * dx - dxx * dy);
//			bool weak = dst0 < -1.f || dst0 > 1.f || dst1 < -1.f || dst1 > 1.f;
//			//float sz = (weak ? -1 : 1) * 2.f * pt->size;
//			float octsub = (dst0 < 0 ? -1 : 1) * (o + fabs(dst0));
//			int ratio = 1 << o;
//			float newo = weak ? o : octsub;
//			float newosub = fabs(newo);
//			float subp = (newo < 0 ? -1 : 1) * (newosub - (int)newosub);
//			pt->y = ratio * (y + dst1);
//			pt->x = ratio * (x + subp);
//			//pt->y = ratio * ((int)(0.5f + pt->y / ratio) + dst1);
//			//pt->x = ratio * ((int)(0.5f + pt->x / ratio) + subp);
//			printf("Y offset: %f, X offset: %f\n", dst1, subp);
//		}

	}


	void Akazer::fastDetect(AkazeData& result, void* tmem, unsigned char* image, int3* owhps, int* osizes, int* offsets)
	{
		// Get address of point counter
		unsigned int* d_point_counter_addr;
		getPointCounter((void**)&d_point_counter_addr);
		CHECK(cudaMemset(d_point_counter_addr, 0, sizeof(unsigned int)));
		setMaxNumPoints(result.max_pts);

		int w, h, p, msz, ms_msz, mstep;
		int* response_map = (int*)tmem;
		float* size_map = (float*)response_map + osizes[0];
		int* layer_map = (int*)(size_map + osizes[0]);

		size_t nbytes = osizes[0] * sizeof(int);
		float minv = -1e6f;
		int* iminv = (int*)&minv;
		CHECK(cudaMemset(layer_map, -1, nbytes));
		CHECK(cudaMemset(response_map, -1E6, nbytes));
		CHECK(cudaMemset(size_map, *iminv, nbytes));

		int* oldnld = NULL;
		int* nldimg = NULL;
		int* smooth = NULL;
		int* flow = NULL;
		int* temp = NULL;
		int* dx = NULL;
		int* dy = NULL;

		float tmax = 0.25f;
		float esigma = soffset;
		float last_etime = 0.5 * soffset * soffset;
		float curr_etime = 0;
		float ttime = 0;
		int naux = 0;
		int oratio = 1;
		int sigma_size = 0;

		float smax = 1.0f;
		if (FEATURE_TYPE == 0 || FEATURE_TYPE == 1 || FEATURE_TYPE == 4 || FEATURE_TYPE == 5)
		{
			smax = 10.0 * sqrtf(2.0f);
		}
		else if (FEATURE_TYPE == 2 || FEATURE_TYPE == 3)
		{
			smax = 12.0 * sqrtf(2.0f);
		}
		std::unique_ptr<float> exptr(new float[max_scale * 2]);
		float* borders = exptr.get();
		float* sizes = borders + max_scale;
		float psz = 10000;
		int neigh = 0;
		//float* threshs = borders + max_scale;

		int ikcontrast = 1;
		int idthreshold = 65;

#ifdef DEBUG_SHOW
		cv::Mat nldshow, detshow, dxshow, dyshow;
		cv::Mat unldshow, udetshow, udxshow, udyshow;
		cv::Mat response_show(owhps[0].y, owhps[0].x, CV_32SC1);
		cv::Mat size_show(owhps[0].y, owhps[0].x, CV_32FC1);
		cv::Mat layer_show(owhps[0].y, owhps[0].x, CV_32SC1);;
#endif // DEBUG_SHOW	

		for (int i = 0; i < noctaves; i++)
		{
			w = owhps[i].x;
			h = owhps[i].y;
			p = owhps[i].z;
			msz = osizes[i];
			ms_msz = msz * max_scale;

#ifdef DEBUG_SHOW
			nldshow.create(h, w, CV_32SC1);
			detshow.create(h, w, CV_32SC1);
			dxshow.create(h, w, CV_32SC1);
			dyshow.create(h, w, CV_32SC1);
#endif // DEBUG_SHOW

			nldimg = (int*)tmem + offsets[i];
			smooth = nldimg + ms_msz;
			flow = smooth + ms_msz;
			temp = flow + ms_msz;
			dx = flow;
			dy = temp;

			// Create nonlinear space for current octave layer
			for (int j = 0; j < max_scale; j++)
			{
				if (j == 0 && i == 0)
				{
					float var = soffset * soffset;
					int ksz = 2 * ceilf((soffset - 0.8f) / 0.3f) + 3;

					fastakaze::hConv2dR2(image, smooth, w, h, p, 1.f);
					fastakaze::hScharrContrast(smooth, temp, ikcontrast, per, w, h, p);
					fastakaze::hLowPass(image, nldimg, w, h, p, var, ksz);
					CHECK(cudaMemcpy(smooth, nldimg, msz * sizeof(int), cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(int) * w, nldimg, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(int) * w, smooth, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(int) * w, temp, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					nldshow.convertTo(unldshow, CV_8UC1);
					detshow.convertTo(udetshow, CV_8UC1);
					dxshow.convertTo(udxshow, CV_8UC1);
#endif // DEBUG_SHOW

					// Compute Hessian Determinant
					sizes[j] = esigma * derivative_factor;
					sigma_size = (int)(esigma * derivative_factor + 0.5f);
					borders[j] = smax * sigma_size;
					fastakaze::hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(int) * w, nldimg, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(int) * w, smooth, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(int) * w, dx, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dyshow.data, sizeof(int) * w, dy, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					nldshow.convertTo(unldshow, CV_8UC1);
					detshow.convertTo(udetshow, CV_8UC1);
					dxshow.convertTo(udxshow, CV_8UC1);
					dyshow.convertTo(udyshow, CV_8UC1);
#endif // DEBUG_SHOW

					continue;
				}

				std::vector<float> tau;
				esigma = soffset * powf(2, (float)j / max_scale + i);
				curr_etime = 0.5f * esigma * esigma;
				ttime = curr_etime - last_etime;
				naux = fed_tau_by_process_time(ttime, 1, tmax, reordering, tau);
				sizes[j] = esigma * derivative_factor / oratio;
				sigma_size = (int)(sizes[j] + 0.5f);
				borders[j] = smax * sigma_size;

				//for (int k = 0; k < naux; k++)
				//{
				//	printf("%f ", tau[k]);
				//}
				//printf("\n");

				if (j == 0)
				{
					ikcontrast = (int)(ikcontrast * 0.75f + 0.5f);
					oldnld = nldimg - mstep;
					fastakaze::hDownWithSmooth(oldnld, nldimg, smooth, owhps[i - 1], owhps[i]);
					fastakaze::hFlow(smooth, flow, diffusivity, ikcontrast, w, h, p);

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(int) * w, nldimg, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(int) * w, smooth, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(int) * w, dx, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					nldshow.convertTo(unldshow, CV_8UC1);
					detshow.convertTo(udetshow, CV_8UC1);
					dxshow.convertTo(udxshow, CV_8UC1);
#endif // DEBUG_SHOW
					for (int k = 0; k < naux; k++)
					{
						fastakaze::hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(int), cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SHOW
						CHECK(cudaMemcpy2D(nldshow.data, sizeof(int) * w, nldimg, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
						nldshow.convertTo(unldshow, CV_8UC1);
#endif // DEBUG_SHOW
					}
				}
				else
				{
					oldnld = nldimg;
					nldimg += msz;
					smooth += msz;
					flow += msz;
					temp += msz;
					dx = flow;
					dy = temp;

					fastakaze::hConv2dR2(oldnld, smooth, w, h, p, 1.f);
					fastakaze::hFlow(smooth, flow, diffusivity, ikcontrast, w, h, p);
					fastakaze::hNldStep(oldnld, flow, nldimg, tau[0], w, h, p);

#ifdef DEBUG_SHOW
					CHECK(cudaMemcpy2D(nldshow.data, sizeof(int) * w, nldimg, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(detshow.data, sizeof(int) * w, smooth, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy2D(dxshow.data, sizeof(int) * w, flow, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
					nldshow.convertTo(unldshow, CV_8UC1);
					detshow.convertTo(udetshow, CV_8UC1);
					dxshow.convertTo(udxshow, CV_8UC1);
#endif // DEBUG_SHOW
					for (int k = 1; k < naux; k++)
					{
						fastakaze::hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(int), cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SHOW
						CHECK(cudaMemcpy2D(nldshow.data, sizeof(int) * w, nldimg, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
						nldshow.convertTo(unldshow, CV_8UC1);
#endif // DEBUG_SHOW						
					}
				}

				fastakaze::hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);

#ifdef DEBUG_SHOW
				CHECK(cudaMemcpy2D(detshow.data, sizeof(int) * w, smooth, sizeof(int) * p, sizeof(int) * w, h, cudaMemcpyDeviceToHost));
				detshow.convertTo(udetshow, CV_8UC1);
#endif // DEBUG_SHOW
				last_etime = curr_etime;
			}

			// Detect keypoints for current octave layer
			int* dets = (int*)tmem + offsets[i] + ms_msz;	// (smooth - ms_msz)
			fastakaze::hCalcExtremaMap(dets, response_map, size_map, layer_map, borders, i, max_scale, idthreshold, w, h, p, owhps[0].z);
			psz = MIN(psz, borders[0] * oratio);
			neigh = MAX(neigh, sigma_size);

			mstep = ms_msz * 4;
			oratio *= 2;
		}

#ifdef DEBUG_SHOW
		CHECK(cudaMemcpy2D(response_show.data, sizeof(int) * owhps[0].x, response_map, sizeof(int) * owhps[0].z, sizeof(int) * owhps[0].x, owhps[0].y, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy2D(size_show.data, sizeof(float) * owhps[0].x, size_map, sizeof(float) * owhps[0].z, sizeof(float) * owhps[0].x, owhps[0].y, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy2D(layer_show.data, sizeof(int) * owhps[0].x, layer_map, sizeof(int) * owhps[0].z, sizeof(int) * owhps[0].x, owhps[0].y, cudaMemcpyDeviceToHost));
#endif // DEBUG_SHOW

		// NMS
		//hNms(result.d_data, response_map, size_map, layer_map, (int)psz, owhps[0].x, owhps[0].y, owhps[0].z);
		fastakaze::hNmsR(result.d_data, response_map, size_map, layer_map, (int)psz, neigh, owhps[0].x, owhps[0].y, owhps[0].z);
		CHECK(cudaMemcpy(&result.num_pts, d_point_counter_addr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		result.num_pts = MIN(result.num_pts, result.max_pts);

		// Refine points
		setOparam(osizes, 5 * noctaves + 1);
		fastakaze::hRefine(result, tmem, noctaves, max_scale);
	}

}