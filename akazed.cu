#include "akazed.h"
#include <device_launch_parameters.h>
#include <memory>


#define X1 256
#define X2 16
#define NBINS 300
#define MAX_SCALE 5
#define MAX_OCTAVE 8
#define MAX_DIST 96
#define FMIN_VAL (-1E6F)
#define IMIN_VAL -1E6

// #define LOG_TIME


__device__ unsigned int d_max_contrast;
__device__ int d_hist[NBINS];
__constant__ float d_extrema_param[MAX_SCALE * 2];
__constant__ int d_max_num_points;
__device__ unsigned int d_point_counter;
__constant__ int d_oparams[MAX_OCTAVE * 5];
__constant__ int comp_idx_1[61 * 8];
__constant__ int comp_idx_2[61 * 8];



void getMaxContrastAddr(void** addr)
{
	CHECK(cudaGetSymbolAddress(addr, d_max_contrast));
}


void setHistogram(const int* h_hist)
{
	CHECK(cudaMemcpyToSymbol(d_hist, h_hist, NBINS * sizeof(int), 0, cudaMemcpyHostToDevice));
}


void setExtremaParam(const float* param, const int n)
{
	CHECK(cudaMemcpyToSymbol(d_extrema_param, param, n * sizeof(float), 0, cudaMemcpyHostToDevice));
}


void getPointCounter(void** addr)
{
	CHECK(cudaGetSymbolAddress(addr, d_point_counter));
}


void setMaxNumPoints(const int num)
{
	CHECK(cudaMemcpyToSymbol(d_max_num_points, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
}


void setOparam(const int* oparams, const int n)
{
	CHECK(cudaMemcpyToSymbol(d_oparams, oparams, n * sizeof(int), 0, cudaMemcpyHostToDevice));
}


void setCompareIndices()
{
	int comp_idx_1_h[61 * 8];
	int comp_idx_2_h[61 * 8];

	int cntr = 0, i = 0, j = 0;
	for (j = 0; j < 4; ++j) 
	{
		for (i = j + 1; i < 4; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j;
			comp_idx_2_h[cntr] = 3 * i;
			cntr++;
		}
	}
	for (j = 0; j < 3; ++j) 
	{
		for (i = j + 1; i < 4; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j + 1;
			comp_idx_2_h[cntr] = 3 * i + 1;
			cntr++;
		}
	}
	for (j = 0; j < 3; ++j) 
	{
		for (i = j + 1; i < 4; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j + 2;
			comp_idx_2_h[cntr] = 3 * i + 2;
			cntr++;
		}
	}

	// 3x3
	for (j = 4; j < 12; ++j) 
	{
		for (i = j + 1; i < 13; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j;
			comp_idx_2_h[cntr] = 3 * i;
			cntr++;
		}
	}
	for (j = 4; j < 12; ++j) 
	{
		for (i = j + 1; i < 13; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j + 1;
			comp_idx_2_h[cntr] = 3 * i + 1;
			cntr++;
		}
	}
	for (j = 4; j < 12; ++j) 
	{
		for (i = j + 1; i < 13; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j + 2;
			comp_idx_2_h[cntr] = 3 * i + 2;
			cntr++;
		}
	}

	// 4x4
	for (j = 13; j < 28; ++j) 
	{
		for (i = j + 1; i < 29; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j;
			comp_idx_2_h[cntr] = 3 * i;
			cntr++;
		}
	}
	for (j = 13; j < 28; ++j) 
	{
		for (i = j + 1; i < 29; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j + 1;
			comp_idx_2_h[cntr] = 3 * i + 1;
			cntr++;
		}
	}
	for (j = 13; j < 28; ++j) 
	{
		for (i = j + 1; i < 29; ++i) 
		{
			comp_idx_1_h[cntr] = 3 * j + 2;
			comp_idx_2_h[cntr] = 3 * i + 2;
			cntr++;
		}
	}

	CHECK(cudaMemcpyToSymbol(comp_idx_1, comp_idx_1_h, 8 * 61 * sizeof(int)));
	CHECK(cudaMemcpyToSymbol(comp_idx_2, comp_idx_2_h, 8 * 61 * sizeof(int)));
}


__inline__ __device__ int borderAdd(const int a, const int b, const int m)
{
	const int c = a + b;
	if (c < m)
	{
		return c;
	}
	return m + m - 2 - c;
}


inline __device__ float dFastAtan2(float y, float x)
{
	const float absx = fabs(x);
	const float absy = fabs(y);
	const float a = __fdiv_rn(min(absx, absy), max(absx, absy));
	const float s = a * a;
	float r = __fmaf_rn(__fmaf_rn(__fmaf_rn(-0.0464964749f, s, 0.15931422f), s, -0.327622764f), s * a, a);
	//float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
	r = (absy > absx ? H_PI - r : r);
	r = (x < 0 ? M_PI - r : r);
	r = (y < 0 ? -r : r);
	return r;
}





namespace akaze
{
	__constant__ float d_lowpass_kernel[21];


	void setLowPassKernel(const float* kernel, const int ksz)
	{
		CHECK(cudaMemcpyToSymbol(d_lowpass_kernel, kernel, ksz * sizeof(float), 0, cudaMemcpyHostToDevice));
	}	



	/* Convolution functions */
	template <int RADIUS>
	__global__ void gConv2d(float* src, float* dst, int width, int height, int pitch)
	{
		__shared__ float sdata[X2 + 2 * RADIUS][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by row
		int ystart = iy * pitch;
		int idx = ystart + ix;
		// int wsubor = width + width - 2;
		int hsubor = height + height - 2;
		int idx0 = idx, idx1 = idx;
		int toy = RADIUS + tiy;
		int br_border = X2 - 1;

		// Middle center
		float wsum = src[idx] * d_lowpass_kernel[0];
		for (int i = 1; i <= RADIUS; i++)
		{	
			// Left
			idx0 = abs(ix - i);
			idx0 += ystart;
			// Right
			idx1 = borderAdd(ix, i, width);
			idx1 += ystart;
			// Weight
			wsum += d_lowpass_kernel[i] * (src[idx0] + src[idx1]);
		}
		sdata[toy][tix] = wsum;

		// Paddding center
		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= RADIUS && tiy > 0)
		{
			at_edge = true;
			new_toy = RADIUS - tiy;	// toy - 2 * tiy
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= br_border && tiy < br_border)
		{
			at_edge = true;
			new_toy = 2 * (br_border + RADIUS) - toy;
			new_iy = borderAdd(iy, 2 * (br_border - tiy), height);
		}
		else if (iy + RADIUS >= height)
		{
			at_edge = true;
			new_toy = toy + RADIUS;
			new_iy = hsubor - (RADIUS + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			int new_idx = new_ystart + ix;
			wsum = src[new_idx] * d_lowpass_kernel[0];
			for (int i = 1; i <= RADIUS; i++)
			{
				// Left
				idx0 = abs(ix - i);
				idx0 += new_ystart;
				// Right
				idx1 = borderAdd(ix, i, width);
				idx1 += new_ystart;
				// Weight
				wsum += d_lowpass_kernel[i] * (src[idx0] + src[idx1]);
			}
			sdata[new_toy][tix] = wsum;
		}
		__syncthreads();

		// Weighted by col
		wsum = sdata[toy][tix] * d_lowpass_kernel[0];
		for (int i = 1; i <= RADIUS; i++)
		{
			wsum += d_lowpass_kernel[i] * (sdata[toy - i][tix] + sdata[toy + i][tix]);
		}
		dst[idx] = wsum;
	}


	__global__ void gConv2dR2(float* src, float* dst, int width, int height, int pitch)
	{
#define MX (X2 - 1)
		__shared__ float sdata[X2 + 2 * 2][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		float* kernel = d_lowpass_kernel;

		// Weighted by row
		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);

		// Middle center
		int toy = tiy + 2;
		sdata[toy][tix] = kernel[0] * src[ystart + ix] + 
			kernel[1] * (src[ystart + ixl1] + src[ystart + ixr1]) +
			kernel[2] * (src[ystart + ixl2] + src[ystart + ixr2]);		

		// Paddding center
		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;	// toy - 2 * tiy
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= MX && tiy < MX)
		{
			at_edge = true;
			new_toy = 2 * (MX + 2) - toy;
			new_iy = borderAdd(iy, 2 * (MX - tiy), height);
		}
		else if (iy + 2 >= height)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_iy = height + height - 2 - (2 + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			sdata[new_toy][tix] = kernel[0] * src[new_ystart + ix] + 
				kernel[1] * (src[new_ystart + ixl1] + src[new_ystart + ixr1]) +
				kernel[2] * (src[new_ystart + ixl2] + src[new_ystart + ixr2]);
		}
		__syncthreads();

		// Weighted by col
		dst[ystart + ix] = kernel[0] * sdata[toy][tix] +
			kernel[1] * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			kernel[2] * (sdata[toy - 2][tix] + sdata[toy + 2][tix]);
	}


	template <int RADIUS>
	__global__ void gConvRow(float* src, float* dst, int width, int height, int pitch)
	{
		__shared__ float sdata[X1 + RADIUS + RADIUS];
		int tid = threadIdx.x;
		int ix = blockIdx.x * blockDim.x + tid;
		int iy = blockIdx.y;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Copy data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tod = RADIUS + tid;
		int right_border = X1 - 1;
		sdata[tod] = src[idx];
		if (tid <= RADIUS && tid > 0)
		{
			int nix = abs(ix - 2 * tid);
			sdata[RADIUS - tid] = src[ystart + nix];
		}
		else if (tod >= right_border && tid < right_border)
		{
			int nix = borderAdd(ix, 2 * (right_border - tid), width);
			sdata[2 * (right_border + RADIUS) - tod] = src[ystart + nix];
		}
		else if (ix + RADIUS >= width)
		{
			int nix = width + width - 2 - (ix + RADIUS);
			sdata[tod + RADIUS] = src[ystart + nix];
		}
		__syncthreads();

		// Weighted by row
		float wsum = sdata[tod] * d_lowpass_kernel[0];
		for (int i = 1; i <= RADIUS; i++)
		{
			wsum += d_lowpass_kernel[i] * (sdata[tod - i] + sdata[tod + i]);
		}
		dst[idx] = wsum;
	}


	template <int RADIUS>
	__global__ void gConvCol(float* src, float* dst, int width, int height, int pitch)
	{
		__shared__ float sdata[X1 + RADIUS + RADIUS];
		int tid = threadIdx.x;
		int iy = blockIdx.x * blockDim.x + tid;
		int ix = blockIdx.y;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Copy data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tod = RADIUS + tid;
		int right_border = X1 - 1;
		sdata[tod] = src[idx];
		if (tid <= RADIUS && tid > 0)
		{
			int niy = abs(iy - 2 * tid);
			sdata[RADIUS - tid] = src[niy * pitch + ix];
		}
		else if (tod >= right_border && tid < right_border)
		{
			int niy = borderAdd(iy, 2 * (right_border - tid), height);
			sdata[2 * (right_border + RADIUS) - tod] = src[niy * pitch + ix];
		}
		else if (iy + RADIUS >= height)
		{
			int niy = height + height - 2 - (iy + RADIUS);
			sdata[tod + RADIUS] = src[niy * pitch + ix];
		}
		__syncthreads();

		// Weighted by row
		float wsum = sdata[tod] * d_lowpass_kernel[0];
		for (int i = 1; i <= RADIUS; i++)
		{
			wsum += d_lowpass_kernel[i] * (sdata[tod - i] + sdata[tod + i]);
		}
		dst[idx] = wsum;
	}


	__global__ void gDownWithSmooth(float* src, float* dst, float* smooth, int3 swhp, int3 dwhp)
	{
		__shared__ float sdata[X2 + 4][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int dix = blockIdx.x * blockDim.x + tix;
		int diy = blockIdx.y * blockDim.y + tiy;
		if (dix >= dwhp.x || diy >= dwhp.y)
		{
			return;
		}
		int six = dix + dix;
		int siy = diy + diy;
		int ystart = siy * swhp.z;
		int toy = tiy + 2;

		// Weighted by row
		int sxes[5] = { abs(six - 4), abs(six - 2), six, borderAdd(six, 2, swhp.x), borderAdd(six, 4, swhp.x) };
		//int syes[5] = { abs(siy - 4), abs(siy - 2), siy, borderAdd(siy, 2, swhp.y), borderAdd(siy, 4, swhp.y) };

		// Current row
		sdata[toy][tix] = d_lowpass_kernel[0] * src[ystart + sxes[2]] +
			d_lowpass_kernel[1] * (src[ystart + sxes[1]] + src[ystart + sxes[3]]) +
			d_lowpass_kernel[2] * (src[ystart + sxes[0]] + src[ystart + sxes[4]]);

		int yborder = X2 - 1;
		int new_toy = toy, new_siy = siy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;	// toy - 2 * tiy
			new_siy = abs(siy - 4 * tiy);
		}
		else if (toy >= yborder && tiy < yborder)
		{
			at_edge = true;
			new_toy = 2 * (yborder + 2) - toy;
			new_siy = borderAdd(siy, 4 * (yborder - tiy), swhp.y);
		}
		else if (siy + 4 >= swhp.y)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_siy = swhp.y + swhp.y - 2 - (4 + siy);
		}

		if (at_edge)
		{
			int new_ystart = new_siy * swhp.z;
			sdata[new_toy][tix] = d_lowpass_kernel[0] * src[new_ystart + sxes[2]] +
				d_lowpass_kernel[1] * (src[new_ystart + sxes[1]] + src[new_ystart + sxes[3]]) +
				d_lowpass_kernel[2] * (src[new_ystart + sxes[0]] + src[new_ystart + sxes[4]]);
		}
		__syncthreads();

		// Weighted by col
		int didx = diy * dwhp.z + dix;
		dst[didx] = src[ystart + six];
		smooth[didx] = d_lowpass_kernel[0] * sdata[toy][tix] +
			d_lowpass_kernel[1] * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			d_lowpass_kernel[2] * (sdata[toy - 2][tix] + sdata[toy + 2][tix]);
	}


	__global__ void gScharrContrastWithMaximum(float* src, float* grad, int width, int height, int pitch)
	{
		__shared__ float sdata[X2 + 2][X2 + 2];
		__shared__ float mdata[X2 * X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int tid = tiy * X2 + tix;
		int ix = blockIdx.x * X2 + tix;
		int iy = blockIdx.y * X2 + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Copy data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tox = tix + 1;
		int toy = tiy + 1;
		int tx2 = tox + 1;
		int ty2 = toy + 1;
		sdata[toy][tox] = src[idx];
		
		if (tix == 0)	// Left
		{
			int nix = abs(ix - 1);
			sdata[toy][0] = src[ystart + nix];
		}
		else if (tox == X2)	// Right
		{
			int nix = borderAdd(ix, 1, width);
			sdata[toy][X2 + 1] = src[ystart + nix];
		}
		else if (tix == 1)
		{
			int nix = abs(ix - 2);
			if (tiy == 1)	// Top-left
			{
				int niy = abs(iy - 2);
				sdata[0][0] = src[niy * pitch + nix];
			}
			else if (toy == X2 - 1) // Bottom-left
			{
				int niy = borderAdd(iy, 2, height);
				sdata[X2 + 1][0] = src[niy * pitch + nix];
			}
		}
		else if (tox == X2 - 1)
		{
			int nix = borderAdd(ix, 2, width);
			if (tiy == 1)	// Top - right
			{
				int niy = abs(iy - 2);
				sdata[0][X2 + 1] = src[niy * pitch + nix];
			}
			else if (toy == X2 - 1)	// Bottom-right
			{
				int niy = borderAdd(iy, 2, height);
				sdata[X2 + 1][X2 + 1] = src[niy * pitch + nix];
			}
		}
		else if (ix + 1 == width) // Right image border
		{
			int nix = width - 2;
			sdata[toy][tx2] = src[ystart + nix];
			if (tiy == 0)
			{
				int niy = abs(iy - 1);
				sdata[0][tx2] = src[niy * pitch + nix];
			}
			else if (toy == X2 || iy + 1 == height)
			{
				int niy = borderAdd(iy, 1, height);
				sdata[ty2][tx2] = src[niy * pitch + nix];
			}
		}

		if (tiy == 0)	// Top
		{
			int niy = abs(iy - 1);
			sdata[0][tox] = src[niy * pitch + ix];
		}
		else if (toy == X2)	// Bottom
		{
			int niy = borderAdd(iy, 1, height);
			sdata[X2 + 1][tox] = src[niy * pitch + ix];
		}
		else if (iy + 1 == height)	// Bottom image border
		{
			int niy = height - 2;
			sdata[ty2][tox] = src[niy * pitch + ix];
			if (tix == 0)
			{
				int nix = abs(ix - 1);
				sdata[ty2][0] = src[niy * pitch + nix];
			}
			else if (tox == X2 || ix + 1 == width)
			{
				int nix = borderAdd(ix, 1, width);
				sdata[ty2][tx2] = src[niy * pitch + nix];
			}
		}

		__syncthreads();

		// Apply Scharr filter and compute gradient
		float dx = 10 * (sdata[toy][tx2] - sdata[toy][tix]) + 3 * (sdata[tiy][tx2] + sdata[ty2][tx2] - sdata[tiy][tix] - sdata[ty2][tix]);
		float dy = 10 * (sdata[ty2][tox] - sdata[tiy][tox]) + 3 * (sdata[ty2][tix] + sdata[ty2][tx2] - sdata[tiy][tix] - sdata[tiy][tx2]);
		mdata[tid] = __fsqrt_rn(dx * dx + dy * dy);
		grad[idx] = mdata[tid];
		__syncthreads();

		// Reduce maximum
		int btn = X2 * X2;
		for (int stride = btn / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride && mdata[tid] < mdata[tid + stride])
			{
				mdata[tid] = mdata[tid + stride];
			}
			__syncthreads();
		}
		if (tid == 0)
		{
			unsigned int* gradi = (unsigned int*)&mdata[0];
			atomicMax(&d_max_contrast, *gradi);
		}
	}


	__global__ void gScharrContrastNaive(float* src, float* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		float dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		float dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		dst[irow1 + ix1] = __fsqrt_rn(dx * dx + dy * dy);
	}


	__global__ void gScharrContrastShared(float* src, float* grad, int width, int height, int pitch)
	{
		__shared__ float sdata[X2 + 2][X2 + 2];
		// __shared__ float mdata[X2 * X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix;
		int iy = blockIdx.y * X2 + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Copy data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tox = tix + 1;
		int toy = tiy + 1;
		int tx2 = tox + 1;
		int ty2 = toy + 1;
		sdata[toy][tox] = src[idx];

		if (tix == 0)	// Left
		{
			int nix = abs(ix - 1);
			sdata[toy][0] = src[ystart + nix];
		}
		else if (tox == X2)	// Right
		{
			int nix = borderAdd(ix, 1, width);
			sdata[toy][X2 + 1] = src[ystart + nix];
		}
		else if (tix == 1)
		{
			int nix = abs(ix - 2);
			if (tiy == 1)	// Top-left
			{
				int niy = abs(iy - 2);
				sdata[0][0] = src[niy * pitch + nix];
			}
			else if (toy == X2 - 1) // Bottom-left
			{
				int niy = borderAdd(iy, 2, height);
				sdata[X2 + 1][0] = src[niy * pitch + nix];
			}
		}
		else if (tox == X2 - 1)
		{
			int nix = borderAdd(ix, 2, width);
			if (tiy == 1)	// Top - right
			{
				int niy = abs(iy - 2);
				sdata[0][X2 + 1] = src[niy * pitch + nix];
			}
			else if (toy == X2 - 1)	// Bottom-right
			{
				int niy = borderAdd(iy, 2, height);
				sdata[X2 + 1][X2 + 1] = src[niy * pitch + nix];
			}
		}
		else if (ix + 1 == width) // Right image border
		{
			int nix = width - 2;
			sdata[toy][tx2] = src[ystart + nix];
			if (tiy == 0)
			{
				int niy = abs(iy - 1);
				sdata[0][tx2] = src[niy * pitch + nix];
			}
			else if (toy == X2 || iy + 1 == height)
			{
				int niy = borderAdd(iy, 1, height);
				sdata[ty2][tx2] = src[niy * pitch + nix];
			}
		}

		if (tiy == 0)	// Top
		{
			int niy = abs(iy - 1);
			sdata[0][tox] = src[niy * pitch + ix];
		}
		else if (toy == X2)	// Bottom
		{
			int niy = borderAdd(iy, 1, height);
			sdata[X2 + 1][tox] = src[niy * pitch + ix];
		}
		else if (iy + 1 == height)	// Bottom image border
		{
			int niy = height - 2;
			sdata[ty2][tox] = src[niy * pitch + ix];
			if (tix == 0)
			{
				int nix = abs(ix - 1);
				sdata[ty2][0] = src[niy * pitch + nix];
			}
			else if (tox == X2 || ix + 1 == width)
			{
				int nix = borderAdd(ix, 1, width);
				sdata[ty2][tx2] = src[niy * pitch + nix];
			}
		}

		__syncthreads();

		// Apply Scharr filter and compute gradient
		float dx = 10 * (sdata[toy][tx2] - sdata[toy][tix]) + 3 * (sdata[tiy][tx2] + sdata[ty2][tx2] - sdata[tiy][tix] - sdata[ty2][tix]);
		float dy = 10 * (sdata[ty2][tox] - sdata[tiy][tox]) + 3 * (sdata[ty2][tix] + sdata[ty2][tx2] - sdata[tiy][tix] - sdata[tiy][tx2]);
		grad[idx] = __fsqrt_rn(dx * dx + dy * dy);
	}


	__inline__ __device__ void sort2vals(float* src, int i, int j)
	{
		if (src[i] < src[j])
		{
			float temp = src[i];
			src[i] = src[j];
			src[j] = temp;
		}
	}


	__global__ void gFindMaxContrast(float* src, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int tid = tiy * X2 + tix;
		int ix = blockIdx.x * X2 + tix;
		int iy = blockIdx.y * X2 + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Reduce maximum
		int idx = iy * pitch + ix;
		for (int stride = X2 * X2 / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				int nid = tid + stride;
				int niy = nid / X2;
				int nix = nid % X2;
				int nidx = niy * pitch + nix;
				sort2vals(src, idx, nidx);
			}
			__syncthreads();
		}
		if (tid == 0)
		{
			unsigned int* gradi = (unsigned int*)&src[idx];
			atomicMax(&d_max_contrast, *gradi);
		}

	}


	__global__ void gFindMaxContrastU4(float* src, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int tid = tiy * X2 + tix;
		int ix0 = blockIdx.x * X2 * 2 + tix;
		int iy0 = blockIdx.y * X2 * 2 + tiy;
		int ix1 = ix0 + X2;
		int iy1 = iy0 + X2;
		if (ix0 >= width || iy0 >= height)
		{
			return;
		}

		// Unroll
		int x0y0 = iy0 * pitch + ix0;
		if (iy1 < height)
		{
			int x0y1 = iy1 * pitch + ix0;
			sort2vals(src, x0y0, x0y1);
			if (ix1 < width)
			{
				int x1y1 = iy1 * pitch + ix1;
				sort2vals(src, x0y0, x1y1);
			}
		}
		if (ix1 < width)
		{
			int x1y0 = iy0 * pitch + ix1;
			sort2vals(src, x0y0, x1y0);
		}

		// Reduce maximum
		for (int stride = X2 * X2 / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				int nid = tid + stride;
				int niy = nid / X2;
				int nix = nid % X2;
				int nidx = niy * pitch + nix;
				sort2vals(src, x0y0, nidx);
			}
			__syncthreads();
		}
		if (tid == 0)
		{
			unsigned int* gradi = (unsigned int*)&src[x0y0];
			atomicMax(&d_max_contrast, *gradi);
		}
	}


	__global__ void gConstrastHist(float* grad, float factor, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix;
		int iy = blockIdx.y * X2 + tiy;
		if (ix >= width && iy >= height)
		{
			return;
		}

		int idx = iy * pitch + ix;
		int hi = __fmul_rz(grad[idx], factor);
		if (hi >= NBINS)
		{
			hi = NBINS - 1;
		}
		atomicAdd(d_hist + hi, 1);
	}


	__global__ void gConstrastHistShared(float* grad, float factor, int width, int height, int pitch)
	{
		__shared__ int shist[NBINS];

		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * 32 + tix;
		int iy = blockIdx.y * 16 + tiy;
		if (ix >= width && iy >= height)
		{
			return;
		}

		// Initialization
		int tid = tiy * 32 + tix;
		if (tid < NBINS)
		{
			shist[tid] = 0;
		}
		__syncthreads();

		// Statistical
		int idx = iy * pitch + ix;
		int hi = __fmul_rz(grad[idx], factor);
		if (hi >= NBINS)
		{
			hi = NBINS - 1;
		}
		atomicAdd(shist + hi, 1);
		__syncthreads();

		// Cumulative
		if (tid < NBINS)
		{
			atomicAdd(d_hist + tid, shist[tid]);
			//d_hist[tid] += shist[tid];
		}
	}


	__global__ void gFlow(float* src, float* dst, DiffusivityType type, float ikc, int width, int height, int pitch)
	{
		__shared__ float sdata[X2 + 2][X2 + 2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		// int tid = tiy * X2 + tix;
		int ix = blockIdx.x * X2 + tix;
		int iy = blockIdx.y * X2 + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Copy data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tox = tix + 1;
		int toy = tiy + 1;
		int tx2 = tox + 1;
		int ty2 = toy + 1;
		sdata[toy][tox] = src[idx];

		if (tix == 0)	// Left
		{
			int nix = abs(ix - 1);
			sdata[toy][0] = src[ystart + nix];
		}
		else if (tox == X2)	// Right
		{
			int nix = borderAdd(ix, 1, width);
			sdata[toy][X2 + 1] = src[ystart + nix];
		}
		else if (tix == 1)
		{
			int nix = abs(ix - 2);
			if (tiy == 1)	// Top-left
			{
				int niy = abs(iy - 2);
				sdata[0][0] = src[niy * pitch + nix];
			}
			else if (toy == X2 - 1) // Bottom-left
			{
				int niy = borderAdd(iy, 2, height);
				sdata[X2 + 1][0] = src[niy * pitch + nix];
			}
		}
		else if (tox == X2 - 1)
		{
			int nix = borderAdd(ix, 2, width);
			if (tiy == 1)	// Top - right
			{
				int niy = abs(iy - 2);
				sdata[0][X2 + 1] = src[niy * pitch + nix];
			}
			else if (toy == X2 - 1)	// Bottom-right
			{
				int niy = borderAdd(iy, 2, height);
				sdata[X2 + 1][X2 + 1] = src[niy * pitch + nix];
			}
		}
		else if (ix + 1 == width) // Right image border
		{
			int nix = width - 2;
			sdata[toy][tx2] = src[ystart + nix];
			if (tiy == 0)
			{
				int niy = abs(iy - 1);
				sdata[0][tx2] = src[niy * pitch + nix];
			}
			else if (toy == X2 || iy + 1 == height)
			{
				int niy = borderAdd(iy, 1, height);
				sdata[ty2][tx2] = src[niy * pitch + nix];
			}
		}

		if (tiy == 0)	// Top
		{
			int niy = abs(iy - 1);
			sdata[0][tox] = src[niy * pitch + ix];
		}
		else if (toy == X2)	// Bottom
		{
			int niy = borderAdd(iy, 1, height);
			sdata[X2 + 1][tox] = src[niy * pitch + ix];
		}
		else if (iy + 1 == height)	// Bottom image border
		{
			int niy = height - 2;
			sdata[ty2][tox] = src[niy * pitch + ix];
			if (tix == 0)
			{
				int nix = abs(ix - 1);
				sdata[ty2][0] = src[niy * pitch + nix];
			}
			else if (tox == X2 || ix + 1 == width)
			{
				int nix = borderAdd(ix, 1, width);
				sdata[ty2][tx2] = src[niy * pitch + nix];
			}
		}

		__syncthreads();

		// Apply Scharr filter and compute gradient
		float dx = 10 * (sdata[toy][tx2] - sdata[toy][tix]) + 3 * (sdata[tiy][tx2] + sdata[ty2][tx2] - sdata[tiy][tix] - sdata[ty2][tix]);
		float dy = 10 * (sdata[ty2][tox] - sdata[tiy][tox]) + 3 * (sdata[ty2][tix] + sdata[ty2][tx2] - sdata[tiy][tix] - sdata[tiy][tx2]);
		float dif2 = ikc * (dx * dx + dy * dy);
		if (type == PM_G1)
		{
			dst[idx] = __expf(-dif2);
		}
		else if (type == PM_G2)
		{
			dst[idx] = 1.f / (1.f + dif2);
		}
		else if (type == WEICKERT)
		{
			dst[idx] = 1.f - __expf(-3.315f / __powf(dif2, 4));
		}
		else
		{
			dst[idx] = 1.f / __fsqrt_rn(1.f + dif2);
		}
	}


	__global__ void gFlowNaive(float* src, float* dst, DiffusivityType type, float ikc, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		float dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		float dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		float dif2 = ikc * (dx * dx + dy * dy);
		if (type == PM_G1)
		{
			dst[irow1 + ix1] = __expf(-dif2);
		}
		else if (type == PM_G2)
		{
			dst[irow1 + ix1] = 1.f / (1.f + dif2);
		}
		else if (type == WEICKERT)
		{
			dst[irow1 + ix1] = 1.f - __expf(-3.315f / __powf(dif2, 4));
		}
		else
		{
			dst[irow1 + ix1] = 1.f / __fsqrt_rn(1.f + dif2);
		}
	}


	__global__ void gNldStep(float* src, float* flow, float* dst, float stepfac, int width, int height, int pitch)
	{
#define P1 (X2 + 1)
#define P2 (X2 + 2)
		__shared__ float simem[P2][P2];
		__shared__ float sfmem[P2][P2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix;
		int iy = blockIdx.y * X2 + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Copy data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tox = tix + 1;
		int toy = tiy + 1;
		int tx2 = tox + 1;
		int ty2 = toy + 1;
		simem[toy][tox] = src[idx];
		sfmem[toy][tox] = flow[idx];

		if (tix == 0)	// Left
		{
			int nidx = ystart + abs(ix - 1);
			simem[toy][0] = src[nidx];
			sfmem[toy][0] = flow[nidx];
		}
		else if (tox == X2)	// Right
		{
			int nidx = ystart + borderAdd(ix, 1, width);
			simem[toy][P1] = src[nidx];
			sfmem[toy][P1] = flow[nidx];
		}
		else if (tix == 1)
		{
			int nix = abs(ix - 2);
			if (tiy == 1)	// Top-left
			{
				int nidx = abs(iy - 2) * pitch + nix;
				simem[0][0] = src[nidx];
				sfmem[0][0] = flow[nidx];
			}
			else if (toy == X2 - 1) // Bottom-left
			{
				int nidx = borderAdd(iy, 2, height) * pitch + nix;
				simem[P1][0] = src[nidx];
				sfmem[P1][0] = flow[nidx];
			}
		}
		else if (tox == X2 - 1)
		{
			int nix = borderAdd(ix, 2, width);
			if (tiy == 1)	// Top - right
			{
				int nidx = abs(iy - 2) * pitch + nix;
				simem[0][P1] = src[nidx];
				sfmem[0][P1] = flow[nidx];
			}
			else if (toy == X2 - 1)	// Bottom-right
			{
				int nidx = borderAdd(iy, 2, height) * pitch + nix;
				simem[P1][P1] = src[nidx];
				sfmem[P1][P1] = flow[nidx];
			}
		}
		else if (ix + 1 == width) // Right image border
		{
			int nix = width - 2;
			int nidx = ystart + nix;
			simem[toy][tx2] = src[nidx];
			sfmem[toy][tx2] = flow[nidx];
			if (tiy == 0)
			{
				nidx = abs(iy - 1) * pitch + nix;
				simem[0][tx2] = src[nidx];
				sfmem[0][tx2] = flow[nidx];
			}
			else if (toy == X2 || iy + 1 == height)
			{
				nidx = borderAdd(iy, 1, height) * pitch + nix;
				simem[ty2][tx2] = src[nidx];
				sfmem[ty2][tx2] = flow[nidx];
			}
		}

		if (tiy == 0)	// Top
		{
			int nidx = abs(iy - 1) * pitch + ix;
			simem[0][tox] = src[nidx];
			sfmem[0][tox] = flow[nidx];
		}
		else if (toy == X2)	// Bottom
		{
			int nidx = borderAdd(iy, 1, height) * pitch + ix;
			simem[P1][tox] = src[nidx];
			sfmem[P1][tox] = flow[nidx];
		}
		else if (iy + 1 == height)	// Bottom image border
		{
			int ystart = (height - 2) * pitch;
			int nidx = ystart + ix;
			simem[ty2][tox] = src[nidx];
			sfmem[ty2][tox] = flow[nidx];
			if (tix == 0)
			{
				nidx = ystart + abs(ix - 1);
				simem[ty2][0] = src[nidx];
				sfmem[ty2][0] = flow[nidx];
			}
			else if (tox == X2 || ix + 1 == width)
			{
				nidx = ystart + borderAdd(ix, 1, width);
				simem[ty2][tx2] = src[nidx];
				sfmem[ty2][tx2] = flow[nidx];
			}
		}

		__syncthreads();

		float step = (sfmem[toy][tox] + sfmem[toy][tx2]) * (simem[toy][tx2] - simem[toy][tox]) +
			(sfmem[toy][tox] + sfmem[toy][tix]) * (simem[toy][tix] - simem[toy][tox]) +
			(sfmem[toy][tox] + sfmem[ty2][tox]) * (simem[ty2][tox] - simem[toy][tox]) +
			(sfmem[toy][tox] + sfmem[tiy][tox]) * (simem[tiy][tox] - simem[toy][tox]);
		dst[idx] = __fmaf_rn(stepfac, step, src[idx]);
	}


	__global__ void gNldStepNaive(float* src, float* flow, float* dst, float stepfac, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - 1);
		int iy0 = abs(iy1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy2 = borderAdd(iy1, 1, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx1 = ystart1 + ix1;
		float step = (flow[idx1] + flow[ystart1 + ix2]) * (src[ystart1 + ix2] - src[idx1]) +
			(flow[idx1] + flow[ystart1 + ix0]) * (src[ystart1 + ix0] - src[idx1]) +
			(flow[idx1] + flow[ystart2 + ix1]) * (src[ystart2 + ix1] - src[idx1]) +
			(flow[idx1] + flow[ystart0 + ix1]) * (src[ystart0 + ix1] - src[idx1]);
		dst[idx1] = __fmaf_rn(stepfac, step, src[idx1]);
	}


	__global__ void gDerivate(float* src, float* dx, float* dy, int step, float fac1, float fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx = ystart1 + ix1;

		float ul = src[ystart0 + ix0];
		float uc = src[ystart0 + ix1];
		float ur = src[ystart0 + ix2];
		float cl = src[ystart1 + ix0];
		// float cc = src[ystart1 + ix1];
		float cr = src[ystart1 + ix2];
		float ll = src[ystart2 + ix0];
		float lc = src[ystart2 + ix1];
		float lr = src[ystart2 + ix2];

		dx[idx] = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
		dy[idx] = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);		
	}


	__global__ void gHessianDeterminant(float* dx, float* dy, float* detd, int step, float fac1, float fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx = ystart1 + ix1;

		int iul = ystart0 + ix0;
		int iuc = ystart0 + ix1;
		int iur = ystart0 + ix2;
		int icl = ystart1 + ix0;
		// int icc = ystart1 + ix1;
		int icr = ystart1 + ix2;
		int ill = ystart2 + ix0;
		int ilc = ystart2 + ix1;
		int ilr = ystart2 + ix2;

		float dxx = fac1 * (dx[iur] + dx[ilr] - dx[iul] - dx[ill]) + fac2 * (dx[icr] - dx[icl]);
		float dxy = fac1 * (dx[ilr] + dx[ill] - dx[iur] - dx[iul]) + fac2 * (dx[ilc] - dx[iuc]);
		float dyy = fac1 * (dy[ilr] + dy[ill] - dy[iur] - dy[iul]) + fac2 * (dy[ilc] - dy[iuc]);

		detd[idx] = dxx * dyy - dxy * dxy;
	}


	__global__ void gCalcExtremaMap(float* dets, float* response_map, float* size_map, int* layer_map, int octave, int max_scale,
		int psz, float threshold, int width, int height, int pitch, int opitch)
	{
		int curr_scale = blockIdx.z;
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		float border = d_extrema_param[curr_scale];
		float size = d_extrema_param[max_scale + curr_scale];

		// Filter outside
		int left_x = (int)(ix - border + 0.5f) - 1;
		int right_x = (int)(ix + border + 0.5f) + 1;
		int up_y = (int)(iy - border + 0.5f) - 1;
		int down_y = (int)(iy + border + 0.5f) + 1;
		if (left_x < 0 || right_x >= width || up_y < 0 || down_y >= height)
		{
			return;
		}

		// Extrema condition
		float* curr_det = dets + curr_scale * height * pitch;
		int idx = iy * pitch + ix;
		float* vp = curr_det + idx;
		float* vp0 = vp - pitch;
		float* vp2 = vp + pitch;
		if (*vp > threshold && *vp > *vp0 && *vp > *vp2 && *vp > *(vp - 1) && *vp > *(vp + 1) &&
			*vp > *(vp0 - 1) && *vp > *(vp0 + 1) && *vp > *(vp2 - 1) && *vp > *(vp2 + 1))
		{
			// The thread may conflict ( But if the minimum execution unit is block, the thread is safe )
			int oix = (ix << octave);
			int oiy = (iy << octave);
			int oidx = oiy * opitch + oix;
			if (response_map[oidx] < *vp)
			{
				response_map[oidx] = *vp;
				size_map[oidx] = size;
				layer_map[oidx] = octave * max_scale + curr_scale;
			}

			//while (true) 
			//{
			//	if (0 == atomicCAS(mutex, 0, 1)) 
			//	{
			//		// **** critical section ****//
			//		if (response_map[oidx] < *vp)
			//		{
			//			response_map[oidx] = *vp;
			//			size_map[oidx] = size;
			//			octave_map[oidx] = octave;
			//		}
			//		__threadfence();
			//		// **** critical section ****//
			//		atomicExch(mutex, 0);
			//		break;
			//	}
			//}
		}
	}


	__global__ void gNms(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int width, int height, int pitch)
	{
		int ix = blockIdx.x * X2 + threadIdx.x + psz;
		int iy = blockIdx.y * X2 + threadIdx.y + psz;
		if (ix + psz >= width || iy + psz >= height)
		{
			return;
		}

		int vidx = iy * pitch + ix;
		if (d_point_counter >= d_max_num_points || 	// Exceed the max number of points limitation
			layer_map[vidx] < 0 || 					// Not a extrema point
			(layer_map[vidx - 1] >= 0 && response_map[vidx] <= response_map[vidx - 1]) ||						// Left
			(layer_map[vidx + 1] >= 0 && response_map[vidx] < response_map[vidx + 1]) ||						// Right
			(layer_map[vidx - pitch] >= 0 && response_map[vidx] <= response_map[vidx - pitch]) ||				// Top
			(layer_map[vidx + pitch] >= 0 && response_map[vidx] < response_map[vidx + pitch]) ||				// Bottom
			(layer_map[vidx - pitch - 1] >= 0 && response_map[vidx] <= response_map[vidx - pitch - 1]) ||		// Left-top
			(layer_map[vidx - pitch + 1] >= 0 && response_map[vidx] < response_map[vidx - pitch + 1]) ||		// Right-top
			(layer_map[vidx + pitch - 1] >= 0 && response_map[vidx] < response_map[vidx + pitch - 1]) ||		// Left-bottom
			(layer_map[vidx + pitch + 1] >= 0 && response_map[vidx] < response_map[vidx + pitch + 1])			// Right-bottom
			)
		{
			return;
		}

		unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
		if (pi < d_max_num_points)
		{
			points[pi].x = ix;
			points[pi].y = iy;
			points[pi].octave = layer_map[vidx];
			points[pi].size = size_map[vidx];
		}
	}


	__global__ void gNmsR(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int r, int width, int height, int pitch)
	{
		extern __shared__ float sdata[];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		if (ix + psz >= width || iy + psz >= height)
		{
			return;
		}

		int sz = X2 + r + r;
		int brb = X2 - 1;

		// Copy Data to shared memory
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int tox = tix + r;
		int toy = tiy + r;
		// int tx2 = tox + r;
		// int ty2 = toy + r;		
		int systart = toy * sz;
		int sidx = systart + tox;
		sdata[sidx] = layer_map[idx] < 0 ? FMIN_VAL : response_map[idx];

		bool v_edge = false, h_edge = false;
		int new_toy = toy, new_tox = tox, new_ix = ix, new_iy = iy, new_idx = idx, new_systart = systart;

		if (tix <= r && tix > 0)	// Left
		{
			v_edge = true;
			new_tox = r - tix;
			new_ix = abs(ix - tix - tix);
			new_idx = ystart + new_ix;
			sdata[systart + new_tox] = layer_map[new_idx] < 0 ? FMIN_VAL : response_map[new_idx];
		}
		else if (tox >= brb && tix < brb)	// Right
		{
			v_edge = true;
			new_tox = 2 * (brb + r) - tox;
			new_ix = borderAdd(ix, 2 * (brb - tix), width);
			new_idx = ystart + new_ix;
			sdata[systart + new_tox] = layer_map[new_idx] < 0 ? FMIN_VAL : response_map[new_idx];
		}

		if (tiy <= r && tiy > 0)	// Top
		{
			h_edge = true;
			new_toy = r - tiy;
			new_iy = abs(iy - tiy - tiy);
			new_idx = new_iy * pitch + ix;
			new_systart = new_toy * sz;
			sdata[new_systart + tox] = layer_map[new_idx] < 0 ? FMIN_VAL : response_map[new_idx];
		}
		else if (toy >= brb && tix < brb)	// Bottom
		{
			h_edge = true;
			new_toy = 2 * (brb + r) - toy;
			new_iy = borderAdd(iy, 2 * (brb - tiy), height);
			new_idx = new_iy * pitch + ix;
			new_systart = new_toy * sz;
			sdata[new_systart + tox] = layer_map[new_idx] < 0 ? FMIN_VAL : response_map[new_idx];
		}

		if (v_edge && h_edge)	// Corner
		{
			new_idx = new_iy * pitch + new_ix;
			sdata[new_systart + new_tox] = layer_map[new_idx] < 0 ? FMIN_VAL : response_map[new_idx];
		}

		__syncthreads();

		if (layer_map[idx] >= 0)
		{
			float fsz = size_map[idx];
			int isz = (int)(fsz + 0.5f);
			int sqsz = fsz * fsz;
			int ii = 0;
			new_systart = (toy - isz) * sz;
			bool to_nms = false;
			for (int i = -isz; i <= isz; i++)
			{
				ii = i * i;
				new_idx = new_systart + tox - isz;
				for (int j = -isz; j <= isz; j++)
				{
					if (i==0 && j==0)
					{
						continue;
					}
					if (ii + j * j < sqsz &&	// Around center
						(sdata[new_idx] > FMIN_VAL && // Also an extrema
						 (sdata[new_idx] > sdata[sidx] || // Larger than center
						  (sdata[new_idx] == sdata[sidx] && i <= 0 && j <= 0)))	// Equal to center but at top-left
						)
					{
						to_nms = true;
					}
					new_idx++;
				}
				if (to_nms)
				{
					break;
				}
				new_systart += sz;
			}
			if (!to_nms)
			{
				unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
				if (pi < d_max_num_points)
				{
					points[pi].x = ix;
					points[pi].y = iy;
					points[pi].octave = layer_map[idx];
					points[pi].size = size_map[idx];
				}
			}
		}
	}


	__global__ void gNmsRNaive(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int r, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		if (ix + psz >= width || iy + psz >= height)
		{
			return;
		}

		int ystart = iy * pitch;
		int idx = ystart + ix;
		if (layer_map[idx] >= 0)
		{
			float fsz = size_map[idx];
			int isz = (int)(fsz + 0.5f);
			int sqsz = fsz * fsz;
			int ii = 0, new_idx = 0;
			int new_systart = (iy - isz) * pitch;
			bool to_nms = false;
			for (int i = -isz; i <= isz; i++)
			{
				ii = i * i;
				new_idx = new_systart + ix - isz;
				for (int j = -isz; j <= isz; j++)
				{
					if (i == 0 && j == 0)
					{
						continue;
					}
					if (ii + j * j < sqsz &&	// Around center
						(response_map[new_idx] > FMIN_VAL && // Also an extrema
							(response_map[new_idx] > response_map[idx] || // Larger than center
								(response_map[new_idx] == response_map[idx] && i <= 0 && j <= 0)))	// Equal to center but at top-left
						)
					{
						to_nms = true;
					}
					new_idx++;
				}
				if (to_nms)
				{
					break;
				}
				new_systart += pitch;
			}
			if (!to_nms)
			{
				unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
				if (pi < d_max_num_points)
				{
					points[pi].x = ix;
					points[pi].y = iy;
					points[pi].octave = layer_map[idx];
					points[pi].size = size_map[idx];
				}
			}
		}
	}

	__global__ void gRefine(AkazePoint* points, float* tmem, int noctaves, int max_scale)
	{
		unsigned int pi = blockIdx.x * X1 + threadIdx.x;
		if (pi >= d_point_counter)
		{
			return;
		}

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		float* det = tmem + offsets[o] + (max_scale + s) * osizes[o];
		int y = (int)pt->y >> o;
		int x = (int)pt->x >> o;
		int idx = y * p + x;
		//int idx = (int)pt->y * p + (int)pt->x;
		float v2 = det[idx] + det[idx];
		float dx = 0.5f * (det[idx + 1] - det[idx - 1]);
		float dy = 0.5f * (det[idx + p] - det[idx - p]);
		float dxx = det[idx + 1] + det[idx - 1] - v2;
		float dyy = det[idx + p] + det[idx - p] - v2;
		float dxy = 0.25f * (det[idx + p + 1] + det[idx - p - 1] - det[idx - p + 1] - det[idx + p - 1]);
		float dd = dxx * dyy - dxy * dxy;
		float idd = dd != 0.f ? 1.f / dd : 0.f;
		float dst0 = idd * (dxy * dy - dyy * dx);
		float dst1 = idd * (dxy * dx - dxx * dy);
		bool weak = dst0 < -1.f || dst0 > 1.f || dst1 < -1.f || dst1 > 1.f;
		if (weak)
		{
			return;
		}
		//float sz = (weak ? -1 : 1) * 2.f * pt->size;
		//float octsub = (dst0 < 0 ? -1 : 1) * (o + fabs(dst0));
		int ratio = 1 << o;
		//int sign = dst0 < 0 ? -1 : 1;
		//float newo = sign * (o + fabs(dst0));
		//float newosub = fabs(newo);
		//float subp = sign * (newosub - (int)newosub);
		pt->y = ratio * (y + dst1);
		pt->x = ratio * (x + dst0);
		//pt->octave = o;
		//pt->size = sz;
	}


	__global__ void gCalcOrient(AkazePoint* points, float* tmem, int noctaves, int max_scale)
	{
		__shared__ float resx[42], resy[42];
		__shared__ float re8x[42], re8y[42];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		if (tix < 42)
		{
			resx[tix] = 0.f;
			resy[tix] = 0.f;
		}
		__syncthreads();

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		float* dxd = tmem + offsets[o] + (max_scale * 2 + s) * osizes[o];
		float* dyd = dxd + max_scale * osizes[o];
		int step = (int)(pt->size + 0.5f);
		int x = (int)(pt->x + 0.5f) >> o;
		int y = (int)(pt->y + 0.5f) >> o;
		int i = (tix & 15) - 6;
		int j = (tix / 16) - 6;
		int r2 = i * i + j * j;
		if (r2 < 36) 
		{
			float gweight = exp(-r2 * 0.08f); // (2.5f * 2.5f * 2.0f));
			int pos = (y + step * j) * p + (x + step * i);
			float dx = gweight * dxd[pos];
			float dy = gweight * dyd[pos];
			float angle = atan2(dy, dx);
			int a = max(min((int)(angle * (21 / M_PI)) + 21, 41), 0);
			atomicAdd(resx + a, dx);
			atomicAdd(resy + a, dy);
		}
		__syncthreads();

		if (tix < 42) 
		{
			re8x[tix] = resx[tix];
			re8y[tix] = resy[tix];
			for (int k = tix + 1; k < tix + 7; k++) 
			{
				re8x[tix] += resx[k < 42 ? k : k - 42];
				re8y[tix] += resy[k < 42 ? k : k - 42];
			}
		}
		__syncthreads();

		if (tix == 0) 
		{
			float maxr = 0.0f;
			int maxk = 0;
			for (int k = 0; k < 42; k++) 
			{
				float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
				if (r > maxr) 
				{
					maxr = r;
					maxk = k;
				}
			}
			float angle = dFastAtan2(re8y[maxk], re8x[maxk]);
			pt->angle = (angle < 0.0f ? angle + 2.0f * M_PI : angle);
		}
	}


	__global__ void gDescribe(AkazePoint* points, float* tmem, float* fdesc, int noctaves, int max_scale, int size2, int size3, int size4)
	{
#define EXTRACT_S 64
		__shared__ float acc_vals[3 * 30 * EXTRACT_S];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		float* acc_vals_im = &acc_vals[0];
		float* acc_vals_dx = &acc_vals[30 * EXTRACT_S];
		float* acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];
		float* vals = fdesc + pi * 3 * 29;

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		float iratio = 1.f / (1 << o);
		int scale = (int)(pt->size + 0.5f);	// ?
		float xf = pt->x * iratio;
		float yf = pt->y * iratio;
		float ang = pt->angle;
		float co = __cosf(ang);
		float si = __sinf(ang);

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		int p = owhps[o].z;

		float* imd = tmem + offsets[o] + s * osizes[o];
		float* dxd = imd + max_scale * osizes[o] * 2;
		float* dyd = dxd + max_scale * osizes[o];
		int winsize = max(3 * size3, 4 * size4);

		for (int i = 0; i < 30; ++i) 
		{
			int j = i * EXTRACT_S + tix;
			acc_vals_im[j] = 0.f;
			acc_vals_dx[j] = 0.f;
			acc_vals_dy[j] = 0.f;
		}
		__syncthreads();

		for (int i = tix; i < winsize * winsize; i += EXTRACT_S) 
		{
			int y = i / winsize;
			int x = i - winsize * y;
			int m = max(x, y);
			if (m >= winsize) 
				continue;
			int l = x - size2;
			int k = y - size2;
			int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
			int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
			int pos = yp * p + xp;
			float im = imd[pos];
			float dx = dxd[pos];
			float dy = dyd[pos];
			float rx = -dx * si + dy * co;
			float ry = dx * co + dy * si;

			if (m < 2 * size2) 
			{
				int x2 = (x < size2 ? 0 : 1);
				int y2 = (y < size2 ? 0 : 1);
				// Add 2x2
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix] += im;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 3 * size3) 
			{
				int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
				int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
				// Add 3x3
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 4 * size4) 
			{
				int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
				int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
				// Add 4x4
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 2] += ry;
			}
		}
		__syncthreads();

		// Reduce stuff
		float acc_reg;
#pragma unroll
		for (int i = 0; i < 15; ++i) 
		{
			// 0..31 takes care of even accs, 32..63 takes care of odd accs
			int offset = 2 * i + (tix < 32 ? 0 : 1);
			int tix_d = tix < 32 ? tix : tix - 32;
			for (int d = 0; d < 90; d += 30) 
			{
				if (tix_d < 32) 
				{
					acc_reg = acc_vals[3 * 30 * tix_d + offset + d] +
						acc_vals[3 * 30 * (tix_d + 32) + offset + d];
					acc_reg += shiftDown(acc_reg, 1);	// __shfl_down(acc_reg, 1);
					acc_reg += shiftDown(acc_reg, 2);	// __shfl_down(acc_reg, 2);
					acc_reg += shiftDown(acc_reg, 4);	// __shfl_down(acc_reg, 4);
					acc_reg += shiftDown(acc_reg, 8);	// __shfl_down(acc_reg, 8);
					acc_reg += shiftDown(acc_reg, 16);	// __shfl_down(acc_reg, 16);
				}
				if (tix_d == 0) 
				{
					acc_vals[offset + d] = acc_reg;
				}
			}
		}

		__syncthreads();

		// Have 29*3 values to store
		// They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
		if (tix < 29) 
		{
			vals[tix] = acc_vals[tix];
			vals[29 + tix] = acc_vals[29 + tix];
			vals[2 * 29 + tix] = acc_vals[2 * 29 + tix];
		}
	}


	__global__ void gDescribe2(AkazePoint* points, float* tmem, int noctaves, int max_scale, int size2, int size3, int size4)
	{
#define EXTRACT_S 64
		__shared__ float acc_vals[3 * 30 * EXTRACT_S];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		float* acc_vals_im = &acc_vals[0];
		float* acc_vals_dx = &acc_vals[30 * EXTRACT_S];
		float* acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		float iratio = 1.f / (1 << o);
		int scale = (int)(pt->size + 0.5f);	// ?
		float xf = pt->x * iratio;
		float yf = pt->y * iratio;
		float ang = pt->angle;
		float co = __cosf(ang);
		float si = __sinf(ang);

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		int p = owhps[o].z;

		float* imd = tmem + offsets[o] + s * osizes[o];
		float* dxd = imd + max_scale * osizes[o] * 2;
		float* dyd = dxd + max_scale * osizes[o];
		int winsize = max(3 * size3, 4 * size4);

		for (int i = 0; i < 30; ++i)
		{
			int j = i * EXTRACT_S + tix;
			acc_vals_im[j] = 0.f;
			acc_vals_dx[j] = 0.f;
			acc_vals_dy[j] = 0.f;
		}
		__syncthreads();

		for (int i = tix; i < winsize * winsize; i += EXTRACT_S)
		{
			int y = i / winsize;
			int x = i - winsize * y;
			int m = max(x, y);
			if (m >= winsize)
				continue;
			int l = x - size2;
			int k = y - size2;
			int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
			int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
			int pos = yp * p + xp;
			float im = imd[pos];
			float dx = dxd[pos];
			float dy = dyd[pos];
			float rx = -dx * si + dy * co;
			float ry = dx * co + dy * si;

			if (m < 2 * size2)
			{
				int x2 = (x < size2 ? 0 : 1);
				int y2 = (y < size2 ? 0 : 1);
				// Add 2x2
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix] += im;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 3 * size3)
			{
				int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
				int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
				// Add 3x3
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 4 * size4)
			{
				int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
				int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
				// Add 4x4
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 2] += ry;
			}
		}
		__syncthreads();

		// Reduce stuff
		float acc_reg;
#pragma unroll
		for (int i = 0; i < 15; ++i)
		{
			// 0..31 takes care of even accs, 32..63 takes care of odd accs
			int offset = 2 * i + (tix < 32 ? 0 : 1);
			int tix_d = tix < 32 ? tix : tix - 32;
			for (int d = 0; d < 90; d += 30)
			{
				if (tix_d < 32)
				{
					acc_reg = acc_vals[3 * 30 * tix_d + offset + d] +
						acc_vals[3 * 30 * (tix_d + 32) + offset + d];
					acc_reg += shiftDown(acc_reg, 1);	// __shfl_down(acc_reg, 1);
					acc_reg += shiftDown(acc_reg, 2);	// __shfl_down(acc_reg, 2);
					acc_reg += shiftDown(acc_reg, 4);	// __shfl_down(acc_reg, 4);
					acc_reg += shiftDown(acc_reg, 8);	// __shfl_down(acc_reg, 8);
					acc_reg += shiftDown(acc_reg, 16);	// __shfl_down(acc_reg, 16);
				}
				if (tix_d == 0)
				{
					acc_vals[offset + d] = acc_reg;
				}
			}
		}

		__syncthreads();

		// Have 29*3 values to store
		// They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
		if (tix < 61)
		{
			unsigned char desc_r = 0;
#pragma unroll
			for (int i = 0; i < (tix == 60 ? 6 : 8); ++i)
			{
				int idx1 = comp_idx_1[tix * 8 + i];
				int idx2 = comp_idx_2[tix * 8 + i];
				desc_r |= (acc_vals[idx1] > acc_vals[idx2] ? 1 : 0) << i;
			}
			pt->features[tix] = desc_r;
		}
	}


	__global__ void gBuildDescriptor(AkazePoint* points, float* fdesc) 
	{
		int pi = blockIdx.x;
		int tix = threadIdx.x;
		AkazePoint* pt = points + pi;

		if (tix < 61) 
		{
			float* curr_fdesc = fdesc + 3 * 29 * pi;
			unsigned char desc_r = 0;

#pragma unroll
			for (int i = 0; i < (tix == 60 ? 6 : 8); ++i) 
			{
				int idx1 = comp_idx_1[tix * 8 + i];
				int idx2 = comp_idx_2[tix * 8 + i];
				desc_r |= (curr_fdesc[idx1] > curr_fdesc[idx2] ? 1 : 0) << i;
			}

			pt->features[tix] = desc_r;
		}
	}


	__global__ void gMatch(AkazePoint* pts1, AkazePoint* pts2, int n1, int n2)
	{
		__shared__ int idx_1st[X2];
		__shared__ int idx_2nd[X2];
		__shared__ int score_1st[X2];
		__shared__ int score_2nd[X2];

		int pi1 = blockIdx.x;
		int tix = threadIdx.x;

		idx_1st[tix] = 0;
		idx_2nd[tix] = 0;
		score_1st[tix] = 512;
		score_2nd[tix] = 512;
		__syncthreads();


		// curent version fixed with popc, still not convinced
		AkazePoint* pt1 = pts1 + pi1;
		unsigned long long* d1i = (unsigned long long*)pt1->features;
		for (int i = 0; i < n2; i += X2) 
		{
			int pi2 = i + tix;
			AkazePoint* pt2 = pts2 + pi2;
			unsigned long long* d2i = (unsigned long long*)pt2->features;
			if (pi2 < n2) 
			{
				// Check d1[p] with d2[i]
				int score = 0;
#pragma unroll
				for (int j = 0; j < 8; ++j) 
				{
					score += __popcll(d1i[j] ^ d2i[j]);
				}
				if (score < score_1st[tix]) 
				{
					score_2nd[tix] = score_1st[tix];
					score_1st[tix] = score;
					idx_2nd[tix] = idx_1st[tix];
					idx_1st[tix] = pi2;
				}
				else if (score < score_2nd[tix]) 
				{
					score_2nd[tix] = score;
					idx_2nd[tix] = pi2;
				}
			}
		}
		__syncthreads();

		// Reduce
		for (int i = X2 / 2; i >= 1; i /= 2) 
		{
			if (tix < i) 
			{
				int nix = tix + i;
				if (score_1st[nix] < score_1st[tix])
				{
					score_2nd[tix] = score_1st[tix];
					score_1st[tix] = score_1st[nix];
					idx_2nd[tix] = idx_1st[tix];
					idx_1st[tix] = idx_1st[nix];
				}
				else if (score_1st[nix] < score_2nd[tix]) 
				{
					score_2nd[tix] = score_1st[nix];
					idx_2nd[tix] = idx_1st[nix];
				}
				if (score_2nd[nix] < score_2nd[tix]) 
				{
					score_2nd[tix] = score_2nd[nix];
					idx_2nd[tix] = idx_2nd[nix];
				}
			}
		}

		if (tix == 0) 
		{
			if (score_1st[0] < score_2nd[0] && score_1st[0] < MAX_DIST)
			{
				AkazePoint* pt2 = pts2 + idx_1st[0];
				pt1->match = idx_1st[0];
				pt1->distance = score_1st[0];
				pt1->match_x = pt2->x;
				pt2->match_y = pt2->y;
			}
			else
			{
				pt1->match = -1;
				pt1->distance = -1;
				pt1->match_x = -1;
				pt1->match_y = -1;
			}
		}
	}


	inline __device__ int dHammingDistance2(unsigned char* f1, unsigned char* f2)
	{
		int dist = 0;
		//unsigned int* v1 = (unsigned int*)f1;
		//unsigned int* v2 = (unsigned int*)f2;
		unsigned long long* v1 = (unsigned long long*)f1;
		unsigned long long* v2 = (unsigned long long*)f2;
		for (int i = 0; i < 8; i++)
		{
			//dist += __popc(f1[i] ^ f2[i]) + __popc(f1[i + 1] ^ f2[i + 1]) + __popc(f1[i + 2] ^ f2[i + 2]) + __popc(f1[i + 3] ^ f2[i + 3]);
			//dist += __popc(v1[i] ^ v2[i]);
			dist += __popcll(v1[i] ^ v2[i]);
		}
		//dist += __popc(f1[60] ^ f2[60]);

		return dist;
	}


	__global__ void gHammingMatch(AkazePoint* points1, AkazePoint* points2, int n1, int n2)
	{
		__shared__ unsigned char ofeat[FLEN];
		__shared__ int distance[X2];
		__shared__ int indice[X2];
		__shared__ int flags[X2];

		unsigned int tid = threadIdx.x;
		unsigned int bid = blockIdx.x;
		if (bid >= n1)
		{
			return;
		}

		AkazePoint* p1 = &points1[bid];
		AkazePoint* p2 = &points2[tid];

		// Compute in base shared memory
		if (tid == 0)
		{
			for (int i = 0; i < FLEN; i++)
			{
				ofeat[i] = p1->features[i];
			}
		}
		__syncthreads();

		distance[tid] = dHammingDistance2(ofeat, p2->features);
		indice[tid] = tid;
		__syncthreads();

		// Compute in template shared memory
		for (int pi = tid + X2; pi < n2; pi += X2)
		{
			// Compute hamming distance
			p2 = &points2[pi];
			int dist = dHammingDistance2(ofeat, p2->features);
			if (dist < distance[tid])
			{
				distance[tid] = dist;
				indice[tid] = pi;
			}
			__syncthreads();
		}

		// Find minimum 
		for (int stride = X2 / 2; stride > 0; stride >>= 1)
		{
			int ntid = tid + stride;
			if (tid < stride && distance[ntid] < distance[tid])
			{
				int temp = distance[tid];
				distance[tid] = distance[ntid];
				distance[ntid] = temp;

				temp = indice[tid];
				indice[tid] = indice[ntid];
				indice[ntid] = temp;
			}
			__syncthreads();
		}

		// Flags for matching condition
		flags[tid] = distance[0] < distance[tid] ? 1 : 0;
		__syncthreads();

		// Sum over
		if (tid < 8)
		{
			volatile int* vsmem = flags;
			//vsmem[tid] += vsmem[tid + 16];
			vsmem[tid] += vsmem[tid + 8];
			vsmem[tid] += vsmem[tid + 4];
			vsmem[tid] += vsmem[tid + 2];
			vsmem[tid] += vsmem[tid + 1];
		}

		// Get result
		if (tid == 0)
		{
			if (flags[0] == (X2 - 1) && distance[0] < MAX_DIST)
			{
				p2 = &points2[indice[0]];
				p1->match = indice[0];	// indice[0];
				p1->distance = distance[0];
				p1->match_x = p2->x;
				p1->match_y = p2->y;
			}
			else
			{
				p1->match = -1;
				p1->distance = -1;
				p1->match_x = -1;
				p1->match_y = -1;
			}
		}
		__syncthreads();
	}




	void hConv2d(float* src, float* dst, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gConv2d<2> %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hConv2d() execution failed\n");
	}


	void hSepConv2d(float* src, float* dst, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		// Much slower than hConv2d
		dim3 block0(X1);
		dim3 grid0((width + X1 - 1) / X1, height);
		gConvRow<2> << <grid0, block0 >> > (src, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gConvRow<2> %f\n", t0);
#endif // LOG_TIME

		dim3 block1(X1);
		dim3 grid1((height + X1 - 1) / X1, width);
		gConvCol<2> << <grid1, block1 >> > (dst, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t1 = timer.read();
		printf("Time of gConvCol<2> %f\n", t1 - t0);
#endif // LOG_TIME

		CheckMsg("hSepConv2d() execution failed\n");
	}


	void createGaussKernel(float var, int radius)
	{
		static float _var = -1.f;
		static int _radius = 0;
		if (abs(_var - var) < 1e-3 && _radius == radius)
		{
			return;
		}

		_var = var;
		_radius = radius;

		const int ksz = radius + 1;
		std::unique_ptr<float> kptr(new float[ksz]);
		float denom = 1.f / (2.f * var);
		float* kernel = kptr.get();
		float ksum = 0;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] = expf(-i * i * denom);
			if (i == 0)
			{
				ksum += kernel[i];
			}
			else
			{
				ksum += kernel[i] + kernel[i];
			}
		}
		ksum = 1 / ksum;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] *= ksum;
		}
		setLowPassKernel(kernel, ksz);
	}


	void hLowPass(float* src, float* dst, int width, int height, int pitch, float var, int ksz)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
		int r = 2;
#endif // LOG_TIME

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		if (ksz <= 5)
		{
#ifdef LOG_TIME
			r = 2;
#endif // LOG_TIME
			createGaussKernel(var, 2);
			gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 7)
		{
#ifdef LOG_TIME
			r = 3;
#endif // LOG_TIME
			createGaussKernel(var, 3);
			gConv2d<3> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 9)
		{
#ifdef LOG_TIME
			r = 4;
#endif // LOG_TIME
			createGaussKernel(var, 4);
			gConv2d<4> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 11)
		{
#ifdef LOG_TIME
			r = 5;
#endif // LOG_TIME
			createGaussKernel(var, 5);
			gConv2d<5> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else
		{
			std::cerr << "Kernels larger than 11 not implemented" << std::endl;
		}

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gConv2d<%d> %f\n", r, t0);
#endif // LOG_TIME
	}


	void hDownWithSmooth(float* src, float* dst, float* smooth, int3 swhp, int3 dwhp)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		createGaussKernel(1.f, 2);
		dim3 block(X2, X2);
		dim3 grid((dwhp.x + X2 - 1) / X2, (dwhp.y + X2 - 1) / X2);
		gDownWithSmooth << <grid, block >> > (src, dst, smooth, swhp, dwhp);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gDownWithSmooth %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hScaleDown() execution failed\n");
	}


	void hScharrContrast(float* src, float* grad, float& kcontrast, float per, int width, int height, int pitch)
	{
		// Compute gradient and find maximum gradient value
		float h_max_contrast = 0.03f;
		unsigned int* d_max_contrast_addr;
		getMaxContrastAddr((void**)&d_max_contrast_addr);
		//CHECK(cudaMemset(d_max_contrast_addr, 0, sizeof(unsigned int)));
		CHECK(cudaMemcpy(d_max_contrast_addr, &h_max_contrast, sizeof(float), cudaMemcpyHostToDevice));

#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gScharrContrastWithMaximum << <grid, block >> > (src, grad, width, height, pitch);
		//gScharrContrastShared << <grid, block >> > (src, grad, width, height, pitch);
		gScharrContrastNaive << <grid, block >> > (src, grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
#endif // LOG_TIME

		//gFindMaxContrast << <grid, block >> > (grad, width, height, pitch);
		dim3 grid1((width / 2 + X2 - 1) / X2, (height / 2 + X2 - 1) / X2);
		gFindMaxContrastU4 << <grid1, block >> > (grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t1 = timer.read();
#endif // LOG_TIME

		CHECK(cudaMemcpy(&h_max_contrast, d_max_contrast_addr, sizeof(float), cudaMemcpyDeviceToHost));
				
		// Initialize Histogram
		int h_hist[NBINS];
		memset(h_hist, 0, NBINS * sizeof(int));
		CHECK(cudaMemcpyToSymbol(d_hist, h_hist, NBINS * sizeof(int), 0, cudaMemcpyHostToDevice));

		// Statistic histogram
		float hfactor = NBINS / h_max_contrast;
		//gConstrastHist << <grid, block >> > (grad, hfactor, width, height, pitch);
		dim3 block2(32, 16);	// Must bigger than NBINS
		dim3 grid2((width + 32 - 1) / 32, (height + 16 - 1) / 16);
		gConstrastHistShared << <grid2, block2 >> > (grad, hfactor, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t2 = timer.read();
		printf("Time of gScharrContrastNaive %f\n", t0);
		printf("Time of gFindMaxContrastU4 %f\n", t1 - t0);
		printf("Time of gConstrastHist %f\n", t2 - t1);
#endif // LOG_TIME

		CHECK(cudaMemcpyFromSymbol(h_hist, d_hist, NBINS * sizeof(int), 0, cudaMemcpyDeviceToHost));

		// Compute contrast threshold
		int thresh = (width * height - h_hist[0]) * per;
		//printf("contrast threshold: %d\n", thresh);	// maybe a little change
		int cumuv = 0;
		int k = 1;
		while (k < NBINS)
		{
			if (cumuv >= thresh)
			{
				break;
			}
			cumuv += h_hist[k];
			k++;
		}
		kcontrast = k / hfactor;

		CheckMsg("hScharrContrast() execution failed\n");
	}


	void hFlow(float* src, float* flow, DiffusivityType type, float kcontrast, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		float ikc = 1.f / (kcontrast * kcontrast);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gFlow << <grid, block >> > (src, flow, type, ikc, width, height, pitch);
		gFlowNaive << <grid, block >> > (src, flow, type, ikc, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gFlow: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hFlow() execution failed\n");
	}


	void hNldStep(float* img, float* flow, float* temp, float step_size, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		float stepfac = 0.5f * step_size;
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gNldStep << <grid, block >> > (img, flow, temp, stepfac, width, height, pitch);
		gNldStepNaive << <grid, block >> > (img, flow, temp, stepfac, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gNldStep: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hNldStep() execution failed\n");
	}


	void hHessianDeterminant(float* src, float* dx, float* dy, int step, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		float w = 10.f / 3.f;
		float fac1 = 1.f / (2.f * (w + 2.f));
		float fac2 = w * fac1;

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gDerivate << <grid, block >> > (src, dx, dy, step, fac1, fac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
#endif // LOG_TIME

		gHessianDeterminant<<<grid, block>>>(dx, dy, src, step, fac1, fac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t1 = timer.read();
		printf("Time of gDerivate: %f\n", t0);
		printf("Time of gHessianDeterminant: %f\n", t1 - t0);
#endif // LOG_TIME
	
		CheckMsg("hHessianDeterminant() execution failed\n");
	}


	void hCalcExtremaMap(float* dets, float* response_map, float* size_map, int* layer_map, float* params,
		int octave, int max_scale, float threshold, int width, int height, int pitch, int opitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		setExtremaParam(params, max_scale * 2);

		int psz = (int)params[0];	// The minimum border
		int depad = psz * 2;

		dim3 block(X2, X2);
		dim3 grid((width - depad + X2 - 1) / X2, (height - depad + X2 - 1) / X2, max_scale);
		gCalcExtremaMap << <grid, block >> > (dets, response_map, size_map, layer_map, octave, max_scale, psz,
			threshold, width, height, pitch, opitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gCalcExtremaMap: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hCalcExtremaMap() execution failed\n");
	}


	void hNms(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		int psz2 = psz + psz;
		dim3 block(X2, X2);
		dim3 grid((width - psz2 + X2 - 1) / X2, (height - psz2 + X2 - 1) / X2);
		gNms << <grid, block >> > (points, response_map, size_map, layer_map, psz, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gNms: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hNms() execution failed\n");
	}


	void hNmsR(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int neigh, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		int psz2 = psz + psz;
		dim3 block(X2, X2);
		dim3 grid((width - psz2 + X2 - 1) / X2, (height - psz2 + X2 - 1) / X2);
		int shared_radius = X2 + 2 * neigh;
		size_t shared_nbytes = shared_radius * shared_radius * sizeof(float);
		// gNmsR << <grid, block, shared_nbytes >> > (points, response_map, size_map, layer_map, psz, neigh, width, height, pitch);
		gNmsRNaive << <grid, block, shared_nbytes >> > (points, response_map, size_map, layer_map, psz, neigh, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gNmsR: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hNmsR() execution failed\n");
	}


	void hRefine(AkazeData& result, float* tmem, int noctaves, int max_scale)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		gRefine << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gRefine: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hRefine() execution failed\n");
	}


	void hCalcOrient(AkazeData& result, float* tmem, int noctaves, int max_scale)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(13 * 16);
		dim3 grid(result.num_pts);
		gCalcOrient << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gCalcOrient: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hCalcOrient() execution failed\n");
	}


	void hDescribe(AkazeData& result, float* tmem, int noctaves, int max_scale, int patsize)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		int size2 = patsize;
		int size3 = ceilf(2.0f * patsize / 3.0f);
		int size4 = ceilf(0.5f * patsize);

		//float* fdesc = NULL;
		//size_t nbytes = 3 * 29 * result.num_pts * sizeof(float);
		//CHECK(cudaMalloc((void**)&fdesc, nbytes));

		dim3 block(64);
		dim3 grid(result.num_pts);

		//gDescribe << <grid, block >> > (result.d_data, tmem, fdesc, noctaves, max_scale, size2, size3, size4);
		//CHECK(cudaDeviceSynchronize());

		//std::unique_ptr<float> hdesc_ptr(new float[3 * 29 * result.num_pts]);
		//std::unique_ptr<int> h_comp_idx_1_ptr(new int[61 * 8]);
		//std::unique_ptr<int> h_comp_idx_2_ptr(new int[61 * 8]);
		//float* hdesc = hdesc_ptr.get();
		//int* h_comp_idx_1 = h_comp_idx_1_ptr.get();
		//int* h_comp_idx_2 = h_comp_idx_2_ptr.get();		
		//CHECK(cudaMemcpy(hdesc, fdesc, nbytes, cudaMemcpyDeviceToHost));
		//CHECK(cudaMemcpyFromSymbol(h_comp_idx_1, comp_idx_1, 61 * 8 * sizeof(int), 0, cudaMemcpyDeviceToHost));
		//CHECK(cudaMemcpyFromSymbol(h_comp_idx_2, comp_idx_2, 61 * 8 * sizeof(int), 0, cudaMemcpyDeviceToHost));
		//for (int i = 0; i < result.num_pts; i++)
		//{
		//	printf("Float Feature %d: ", i);
		//	for (int j = 0; j < 3 * 29; j++)
		//	{
		//		printf("%f ", hdesc[j]);
		//	}
		//	printf("\n");
		//	float* curr_fdesc = hdesc + 3 * 29 * i;
		//	
		//	printf("UChar Feature %d: ", i);
		//	for (int j = 0; j < 61; j++)
		//	{
		//		unsigned char desc_r = 0;
		//		for (int k = 0; k < (j == 60 ? 6 : 8); ++k)
		//		{
		//			int idx1 = h_comp_idx_1[j * 8 + k];
		//			int idx2 = h_comp_idx_2[j * 8 + k];
		//			desc_r |= (curr_fdesc[idx1] > curr_fdesc[idx2] ? 1 : 0) << k;
		//		}
		//		printf("%d ", (int)desc_r);
		//	}
		//	printf("\n");
		//}

		//gBuildDescriptor << <grid, block >> > (result.d_data, fdesc);
		//CHECK(cudaDeviceSynchronize());

		//unsigned char* h_ptr = result.h_data[0].features;
		//unsigned char* d_ptr = result.d_data[0].features;
		//CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), FLEN * sizeof(unsigned char), result.num_pts, cudaMemcpyDeviceToHost));
		//for (int i = 0; i < result.num_pts; i++)
		//{
		//	akaze::AkazePoint* pt = result.h_data + i;
		//	printf("Feature %d: ", i);
		//	for (int j = 0; j < FLEN; j++)
		//	{
		//		printf("%d ", (int)(pt->features[j]));
		//	}
		//	printf("\n");
		//}

		gDescribe2 << <grid, block >> > (result.d_data, tmem, noctaves, max_scale, size2, size3, size4);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of descriptors computation: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hDescribe() execution failed\n");
	}


	void hMatch(AkazeData& result1, AkazeData& result2)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(X2);
		dim3 grid(result1.num_pts);
		// gMatch << <grid, block >> > (result1.d_data, result2.d_data, result1.num_pts, result2.num_pts);
		gHammingMatch << <grid, block >> > (result1.d_data, result2.d_data, result1.num_pts, result2.num_pts);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of keypoints matching: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hMatch() execution failed\n");
	}

}


namespace fastakaze
{
	__constant__ int d_lowpass_kernel[21];


	__global__ void gConv2dR2(unsigned char* src, int* dst, int width, int height, int pitch)
	{
#define MX (X2 - 1)
		__shared__ int sdata[X2 + 2 * 2][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		int k0 = d_lowpass_kernel[0];
		int k1 = d_lowpass_kernel[1];
		int k2 = d_lowpass_kernel[2];

		// Weighted by row
		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);

		// Middle center
		int toy = tiy + 2;
		sdata[toy][tix] = (k0 * (int)src[ystart + ix] +
			k1 * ((int)src[ystart + ixl1] + (int)src[ystart + ixr1]) +
			k2 * ((int)src[ystart + ixl2] + (int)src[ystart + ixr2])) >> 16;

		// Paddding center
		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;	// toy - 2 * tiy
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= MX && tiy < MX)
		{
			at_edge = true;
			new_toy = 2 * (MX + 2) - toy;
			new_iy = borderAdd(iy, 2 * (MX - tiy), height);
		}
		else if (iy + 2 >= height)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_iy = height + height - 2 - (2 + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			sdata[new_toy][tix] = (k0 * (int)src[new_ystart + ix] +
				k1 * ((int)src[new_ystart + ixl1] + (int)src[new_ystart + ixr1]) +
				k2 * ((int)src[new_ystart + ixl2] + (int)src[new_ystart + ixr2])) >> 16;
		}
		__syncthreads();

		// Weighted by col
		dst[ystart + ix] = (k0 * sdata[toy][tix] +
			k1 * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			k2 * (sdata[toy - 2][tix] + sdata[toy + 2][tix])) >> 16;
	}


	__global__ void gConv2dR2Row(unsigned char* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by row
		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);
		dst[ystart + ix] = (d_lowpass_kernel[0] * (int)src[ystart + ix] +
			d_lowpass_kernel[1] * ((int)src[ystart + ixl1] + (int)src[ystart + ixr1]) +
			d_lowpass_kernel[2] * ((int)src[ystart + ixl2] + (int)src[ystart + ixr2])) >> 16;
	}


	__global__ void gConv2dR2Row(int* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by row
		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);
		dst[ystart + ix] = (d_lowpass_kernel[0] * src[ystart + ix] +
			d_lowpass_kernel[1] * (src[ystart + ixl1] + src[ystart + ixr1]) +
			d_lowpass_kernel[2] * (src[ystart + ixl2] + src[ystart + ixr2])) >> 16;
	}


	__global__ void gConv2dR2Col(int* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by col
		int iyt2 = abs(iy - 2);
		int iyt1 = abs(iy - 1);
		int iyb1 = borderAdd(iy, 1, height);
		int iyb2 = borderAdd(iy, 2, height);
		dst[iy * pitch + ix] = (d_lowpass_kernel[0] * src[iy * pitch + ix] +
			d_lowpass_kernel[1] * (src[iyt1 * pitch + ix] + src[iyb1 * pitch + ix]) +
			d_lowpass_kernel[2] * (src[iyt2 * pitch + ix] + src[iyb2 * pitch + ix])) >> 16;
	}


	__global__ void gConv2dR2(int* src, int* dst, int width, int height, int pitch)
	{
#define MX (X2 - 1)
		__shared__ int sdata[X2 + 2 * 2][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		int k0 = d_lowpass_kernel[0];
		int k1 = d_lowpass_kernel[1];
		int k2 = d_lowpass_kernel[2];

		// Weighted by row
		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);

		// Middle center
		int toy = tiy + 2;
		sdata[toy][tix] = (k0 * src[ystart + ix] +
			k1 * (src[ystart + ixl1] + src[ystart + ixr1]) +
			k2 * (src[ystart + ixl2] + src[ystart + ixr2])) >> 16;

		// Paddding center
		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;	// toy - 2 * tiy
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= MX && tiy < MX)
		{
			at_edge = true;
			new_toy = 2 * (MX + 2) - toy;
			new_iy = borderAdd(iy, 2 * (MX - tiy), height);
		}
		else if (iy + 2 >= height)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_iy = height + height - 2 - (2 + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			sdata[new_toy][tix] = (k0 * src[new_ystart + ix] +
				k1 * (src[new_ystart + ixl1] + src[new_ystart + ixr1]) +
				k2 * (src[new_ystart + ixl2] + src[new_ystart + ixr2])) >> 16;
		}
		__syncthreads();

		// Weighted by col
		dst[ystart + ix] = (k0 * sdata[toy][tix] +
			k1 * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			k2 * (sdata[toy - 2][tix] + sdata[toy + 2][tix])) >> 16;
	}


	template <int RADIUS>
	__global__ void gConv2d(unsigned char* src, int* dst, int width, int height, int pitch)
	{
		__shared__ int sdata[X2 + 2 * RADIUS][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by row
		int ystart = iy * pitch;
		int idx = ystart + ix;
		// int wsubor = width + width - 2;
		int hsubor = height + height - 2;
		int idx0 = idx, idx1 = idx;
		int toy = RADIUS + tiy;
		int br_border = X2 - 1;

		// Middle center
		int wsum = d_lowpass_kernel[0] * (int)src[idx];
		for (int i = 1; i <= RADIUS; i++)
		{
			// Left
			idx0 = abs(ix - i);
			idx0 += ystart;
			// Right
			idx1 = borderAdd(ix, i, width);
			idx1 += ystart;
			// Weight
			wsum += d_lowpass_kernel[i] * ((int)src[idx0] + (int)src[idx1]);
		}
		sdata[toy][tix] = wsum >> 16;

		// Paddding center
		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= RADIUS && tiy > 0)
		{
			at_edge = true;
			new_toy = RADIUS - tiy;	// toy - 2 * tiy
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= br_border && tiy < br_border)
		{
			at_edge = true;
			new_toy = 2 * (br_border + RADIUS) - toy;
			new_iy = borderAdd(iy, 2 * (br_border - tiy), height);
		}
		else if (iy + RADIUS >= height)
		{
			at_edge = true;
			new_toy = toy + RADIUS;
			new_iy = hsubor - (RADIUS + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			int new_idx = new_ystart + ix;
			wsum = d_lowpass_kernel[0] * (int)src[new_idx];
			for (int i = 1; i <= RADIUS; i++)
			{
				// Left
				idx0 = abs(ix - i);
				idx0 += new_ystart;
				// Right
				idx1 = borderAdd(ix, i, width);
				idx1 += new_ystart;
				// Weight
				wsum += d_lowpass_kernel[i] * ((int)src[idx0] + (int)src[idx1]);
			}
			sdata[new_toy][tix] = wsum >> 16;
		}
		__syncthreads();

		// Weighted by col
		wsum = d_lowpass_kernel[0] * sdata[toy][tix];
		for (int i = 1; i <= RADIUS; i++)
		{
			wsum += d_lowpass_kernel[i] * (sdata[toy - i][tix] + sdata[toy + i][tix]);
		}
		dst[idx] = wsum >> 16;
	}


	template <int RADIUS>
	__global__ void gConv2dRow(unsigned char* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by row
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int ix0, ix1;

		// Middle center
		int wsum = d_lowpass_kernel[0] * (int)src[idx];
		for (int i = 1; i <= RADIUS; i++)
		{
			// Left
			ix0 = abs(ix - i);
			// Right
			ix1 = borderAdd(ix, i, width);
			// Weight
			wsum += d_lowpass_kernel[i] * ((int)src[ystart + ix0] + (int)src[ystart + ix1]);
		}
		dst[idx] = wsum >> 16;
	}


	template <int RADIUS>
	__global__ void gConv2dCol(int* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		// Weighted by row
		int ystart = iy * pitch;
		int idx = ystart + ix;
		int iy0, iy1;

		// Middle center
		int wsum = d_lowpass_kernel[0] * src[idx];
		for (int i = 1; i <= RADIUS; i++)
		{
			// Left
			iy0 = abs(iy - i);
			// Right
			iy1 = borderAdd(iy, i, width);
			// Weight
			wsum += d_lowpass_kernel[i] * (src[iy0 * pitch + ix] + src[iy1 * pitch + ix]);
		}
		dst[idx] = wsum >> 16;
	}


	__global__ void gDownWithSmooth(int* src, int* dst, int* smooth, int3 swhp, int3 dwhp)
	{
		__shared__ int sdata[X2 + 4][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int dix = blockIdx.x * blockDim.x + tix;
		int diy = blockIdx.y * blockDim.y + tiy;
		if (dix >= dwhp.x || diy >= dwhp.y)
		{
			return;
		}
		int six = dix + dix;
		int siy = diy + diy;
		int ystart = siy * swhp.z;
		int toy = tiy + 2;

		// Weighted by row
		int sxes[5] = { abs(six - 4), abs(six - 2), six, borderAdd(six, 2, swhp.x), borderAdd(six, 4, swhp.x) };
		//int syes[5] = { abs(siy - 4), abs(siy - 2), siy, borderAdd(siy, 2, swhp.y), borderAdd(siy, 4, swhp.y) };

		// Current row
		sdata[toy][tix] = (d_lowpass_kernel[0] * src[ystart + sxes[2]] +
			d_lowpass_kernel[1] * (src[ystart + sxes[1]] + src[ystart + sxes[3]]) +
			d_lowpass_kernel[2] * (src[ystart + sxes[0]] + src[ystart + sxes[4]])) >> 16;

		int yborder = X2 - 1;
		int new_toy = toy, new_siy = siy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;	// toy - 2 * tiy
			new_siy = abs(siy - 4 * tiy);
		}
		else if (toy >= yborder && tiy < yborder)
		{
			at_edge = true;
			new_toy = 2 * (yborder + 2) - toy;
			new_siy = borderAdd(siy, 4 * (yborder - tiy), swhp.y);
		}
		else if (siy + 4 >= swhp.y)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_siy = swhp.y + swhp.y - 2 - (4 + siy);
		}

		if (at_edge)
		{
			int new_ystart = new_siy * swhp.z;
			sdata[new_toy][tix] = (d_lowpass_kernel[0] * src[new_ystart + sxes[2]] +
				d_lowpass_kernel[1] * (src[new_ystart + sxes[1]] + src[new_ystart + sxes[3]]) +
				d_lowpass_kernel[2] * (src[new_ystart + sxes[0]] + src[new_ystart + sxes[4]])) >> 16;
		}
		__syncthreads();

		// Weighted by col
		int didx = diy * dwhp.z + dix;
		dst[didx] = src[ystart + six];
		smooth[didx] = (d_lowpass_kernel[0] * sdata[toy][tix] +
			d_lowpass_kernel[1] * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			d_lowpass_kernel[2] * (sdata[toy - 2][tix] + sdata[toy + 2][tix])) >> 16;
	}


	__global__ void gScharrContrastNaive(int* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		int dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		int dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		dst[irow1 + ix1] = (int)(__fsqrt_rn(dx * dx + dy * dy) + 0.5f);
	}


	__inline__ __device__ void sort2vals(int* src, int i, int j)
	{
		if (src[i] < src[j])
		{
			int temp = src[i];
			src[i] = src[j];
			src[j] = temp;
		}
	}


	__global__ void gFindMaxContrastU4(int* src, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int tid = tiy * X2 + tix;
		int ix0 = blockIdx.x * X2 * 2 + tix;
		int iy0 = blockIdx.y * X2 * 2 + tiy;
		int ix1 = ix0 + X2;
		int iy1 = iy0 + X2;
		if (ix0 >= width || iy0 >= height)
		{
			return;
		}

		// Unroll
		int x0y0 = iy0 * pitch + ix0;
		if (iy1 < height)
		{
			int x0y1 = iy1 * pitch + ix0;
			sort2vals(src, x0y0, x0y1);
			if (ix1 < width)
			{
				int x1y1 = iy1 * pitch + ix1;
				sort2vals(src, x0y0, x1y1);
			}
		}
		if (ix1 < width)
		{
			int x1y0 = iy0 * pitch + ix1;
			sort2vals(src, x0y0, x1y0);
		}

		// Reduce maximum
		for (int stride = X2 * X2 / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				int nid = tid + stride;
				int niy = nid / X2;
				int nix = nid % X2;
				int nidx = niy * pitch + nix;
				sort2vals(src, x0y0, nidx);
			}
			__syncthreads();
		}
		if (tid == 0)
		{
			atomicMax(&d_max_contrast, src[x0y0]);
		}
	}


	__global__ void gConstrastHistShared(int* grad, int factor, int width, int height, int pitch)
	{
		__shared__ int shist[NBINS];

		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * 32 + tix;
		int iy = blockIdx.y * 16 + tiy;
		if (ix >= width && iy >= height)
		{
			return;
		}

		// Initialization
		int tid = tiy * 32 + tix;
		if (tid < NBINS)
		{
			shist[tid] = 0;
		}
		__syncthreads();

		// Statistical
		int idx = iy * pitch + ix;
		//int hi = (int)(((long long)grad[idx] * factor) >> 32);
		//int hi = (int)(grad[idx] * factor + 0.5f);
		int hi = (grad[idx] * factor) >> 16;
		if (hi >= NBINS)
		{
			hi = NBINS - 1;
		}
		atomicAdd(shist + hi, 1);
		__syncthreads();

		// Cumulative
		if (tid < NBINS)
		{
			atomicAdd(d_hist + tid, shist[tid]);
			//d_hist[tid] += shist[tid];
		}
	}


	__global__ void gDerivate(int* src, int* dx, int* dy, int step, int fac1, int fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int idx = iy1 * pitch + ix1;
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;

		int ul = src[ystart0 + ix0];
		int uc = src[ystart0 + ix1];
		int ur = src[ystart0 + ix2];
		int cl = src[ystart1 + ix0];
		// int cc = src[ystart1 + ix1];
		int cr = src[ystart1 + ix2];
		int ll = src[ystart2 + ix0];
		int lc = src[ystart2 + ix1];
		int lr = src[ystart2 + ix2];

		dx[idx] = (fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl)) >> 16;
		dy[idx] = (fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc)) >> 16;
	}


	__global__ void gHessianDeterminant(int* dx, int* dy, int* detd, int step, int fac1, int fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int idx = iy1 * pitch + ix1;
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;

		int iul = ystart0 + ix0;
		int iuc = ystart0 + ix1;
		int iur = ystart0 + ix2;
		int icl = ystart1 + ix0;
		// int icc = ystart1 + ix1;
		int icr = ystart1 + ix2;
		int ill = ystart2 + ix0;
		int ilc = ystart2 + ix1;
		int ilr = ystart2 + ix2;

		int dxx = (fac1 * (dx[iur] + dx[ilr] - dx[iul] - dx[ill]) + fac2 * (dx[icr] - dx[icl])) >> 16;
		int dxy = (fac1 * (dx[ilr] + dx[ill] - dx[iur] - dx[iul]) + fac2 * (dx[ilc] - dx[iuc])) >> 16;
		int dyy = (fac1 * (dy[ilr] + dy[ill] - dy[iur] - dy[iul]) + fac2 * (dy[ilc] - dy[iuc])) >> 16;

		detd[idx] = dxx * dyy - dxy * dxy;
	}


	__global__ void gFlowNaive(int* src, int* dst, akaze::DiffusivityType type, float ikc, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		int dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		int dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		float dif2 = (dx * dx + dy * dy) * ikc;
		//printf("dif2 = %f\n", dif2);
		if (type == akaze::PM_G1)
		{
			dst[irow1 + ix1] = (int)(__expf(-dif2) * 65536 + 0.5f);
		}
		else if (type == akaze::PM_G2)
		{
			dst[irow1 + ix1] = (int)(1.f / (1.f + dif2) * 65536 + 0.5f);
		}
		else if (type == akaze::WEICKERT)
		{
			dst[irow1 + ix1] = (int)((1.f - __expf(-3.315f / __powf(dif2, 4))) * 65536 + 0.5f);
		}
		else
		{
			dst[irow1 + ix1] = (int)(1.f / __fsqrt_rn(1.f + dif2) * 65536 + 0.5f);
		}
	}


	__global__ void gNldStepNaive(int* src, int* flow, int* dst, int stepfac, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - 1);
		int iy0 = abs(iy1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy2 = borderAdd(iy1, 1, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx1 = ystart1 + ix1;
		int step = ((flow[idx1] + flow[ystart1 + ix2]) * (src[ystart1 + ix2] - src[idx1]) +
			(flow[idx1] + flow[ystart1 + ix0]) * (src[ystart1 + ix0] - src[idx1]) +
			(flow[idx1] + flow[ystart2 + ix1]) * (src[ystart2 + ix1] - src[idx1]) +
			(flow[idx1] + flow[ystart0 + ix1]) * (src[ystart0 + ix1] - src[idx1])) >> 16;
		dst[idx1] = ((stepfac * step) >> 16) + src[idx1];
		//dst[idx1] = __fmaf_rn(stepfac, step, src[idx1]);
	}


	__global__ void gCalcExtremaMap(int* dets, int* response_map, float* size_map, int* layer_map, int octave, int max_scale,
		int psz, int threshold, int width, int height, int pitch, int opitch)
	{
		int curr_scale = blockIdx.z;
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		float border = d_extrema_param[curr_scale];
		float size = d_extrema_param[max_scale + curr_scale];

		// Filter outside
		int left_x = (int)(ix - border + 0.5f) - 1;
		int right_x = (int)(ix + border + 0.5f) + 1;
		int up_y = (int)(iy - border + 0.5f) - 1;
		int down_y = (int)(iy + border + 0.5f) + 1;
		if (left_x < 0 || right_x >= width || up_y < 0 || down_y >= height)
		{
			return;
		}

		// Extrema condition
		int* curr_det = dets + curr_scale * height * pitch;
		int idx = iy * pitch + ix;
		int* vp = curr_det + idx;
		int* vp0 = vp - pitch;
		int* vp2 = vp + pitch;
		if (*vp > threshold && *vp > *vp0 && *vp > *vp2 && *vp > *(vp - 1) && *vp > *(vp + 1) &&
			*vp > *(vp0 - 1) && *vp > *(vp0 + 1) && *vp > *(vp2 - 1) && *vp > *(vp2 + 1))
		{
			// The thread may conflict ( But if the minimum execution unit is block, the thread is safe )
			int oix = (ix << octave);
			int oiy = (iy << octave);
			int oidx = oiy * opitch + oix;
			if (response_map[oidx] < *vp)
			{
				response_map[oidx] = *vp;
				size_map[oidx] = size;
				layer_map[oidx] = octave * max_scale + curr_scale;
			}

			//while (true) 
			//{
			//	if (0 == atomicCAS(mutex, 0, 1)) 
			//	{
			//		// **** critical section ****//
			//		if (response_map[oidx] < *vp)
			//		{
			//			response_map[oidx] = *vp;
			//			size_map[oidx] = size;
			//			octave_map[oidx] = octave;
			//		}
			//		__threadfence();
			//		// **** critical section ****//
			//		atomicExch(mutex, 0);
			//		break;
			//	}
			//}
		}
	}


	__global__ void gNmsRNaive(akaze::AkazePoint* points, int* response_map, float* size_map, int* layer_map, int psz, int r, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		if (ix + psz >= width || iy + psz >= height)
		{
			return;
		}

		int ystart = iy * pitch;
		int idx = ystart + ix;
		if (layer_map[idx] >= 0)
		{
			float fsz = size_map[idx];
			int isz = (int)(fsz + 0.5f);
			int sqsz = fsz * fsz;
			int ii = 0, new_idx = 0;
			int new_systart = (iy - isz) * pitch;
			bool to_nms = false;
			for (int i = -isz; i <= isz; i++)
			{
				ii = i * i;
				new_idx = new_systart + ix - isz;
				for (int j = -isz; j <= isz; j++)
				{
					if (i == 0 && j == 0)
					{
						continue;
					}
					if (ii + j * j < sqsz &&	// Around center
						(response_map[new_idx] > IMIN_VAL && // Also an extrema
							(response_map[new_idx] > response_map[idx] || // Larger than center
								(response_map[new_idx] == response_map[idx] && i <= 0 && j <= 0)))	// Equal to center but at top-left
						)
					{
						to_nms = true;
					}
					new_idx++;
				}
				if (to_nms)
				{
					break;
				}
				new_systart += pitch;
			}
			if (!to_nms && d_point_counter < d_max_num_points)
			{
				unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
				if (pi < d_max_num_points)
				{
					points[pi].x = ix;
					points[pi].y = iy;
					points[pi].octave = layer_map[idx];
					points[pi].size = size_map[idx];
				}
			}
		}
	}


	__global__ void gRefine(akaze::AkazePoint* points, void* tmem, int noctaves, int max_scale)
	{
		unsigned int pi = blockIdx.x * X1 + threadIdx.x;
		if (pi >= d_point_counter)
		{
			return;
		}

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		akaze::AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		int* det = (int*)tmem + offsets[o] + (max_scale + s) * osizes[o];
		int y = (int)pt->y >> o;
		int x = (int)pt->x >> o;
		int idx = y * p + x;
		//int idx = (int)pt->y * p + (int)pt->x;
		int v2 = det[idx] + det[idx];
		int dx = (det[idx + 1] - det[idx - 1]) >> 1;
		int dy = (det[idx + p] - det[idx - p]) >> 1;
		int dxx = det[idx + 1] + det[idx - 1] - v2;
		int dyy = det[idx + p] + det[idx - p] - v2;
		int dxy = (det[idx + p + 1] + det[idx - p - 1] - det[idx - p + 1] - det[idx + p - 1]) >> 2;
		int dd = dxx * dyy - dxy * dxy;
		float idd = dd != 0 ? (1.f / dd) : 0.f;
		float dst0 = idd * (dxy * dy - dyy * dx);
		float dst1 = idd * (dxy * dx - dxx * dy);
		if (dst0 < -1.f || dst0 > 1.f || dst1 < -1.f || dst1 > 1.f)
		{
			return;
		}
		//float sz = (weak ? -1 : 1) * 2.f * pt->size;
		//float octsub = (dst0 < 0 ? -1 : 1) * (o + fabs(dst0));
		int ratio = 1 << o;
		//int sign = dst0 < 0 ? -1 : 1;
		//float newo = sign * (o + fabs(dst0));
		//float newosub = fabs(newo);
		//float subp = sign * (newosub - (int)newosub);
		pt->y = ratio * (y + dst1);
		pt->x = ratio * (x + dst0);
		//pt->octave = o;
		//pt->size = sz;
	}


	__global__ void gCalcOrient(akaze::AkazePoint* points, void* tmem, int noctaves, int max_scale)
	{
		__shared__ float resx[42], resy[42];
		__shared__ float re8x[42], re8y[42];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		if (tix < 42)
		{
			resx[tix] = 0.f;
			resy[tix] = 0.f;
		}
		__syncthreads();

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		akaze::AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		int* dxd = (int*)tmem + offsets[o] + (max_scale * 2 + s) * osizes[o];
		int* dyd = dxd + max_scale * osizes[o];
		int step = (int)(pt->size + 0.5f);
		int x = (int)(pt->x + 0.5f) >> o;
		int y = (int)(pt->y + 0.5f) >> o;
		int i = (tix & 15) - 6;
		int j = (tix / 16) - 6;
		int r2 = i * i + j * j;
		if (r2 < 36)
		{
			float gweight = __expf(-r2 * 0.08f); // (2.5f * 2.5f * 2.0f));
			int pos = (y + step * j) * p + (x + step * i);
			float dx = gweight * dxd[pos];
			float dy = gweight * dyd[pos];
			float angle = dFastAtan2(dy, dx);
			int a = max(min((int)(angle * (21 / M_PI)) + 21, 41), 0);
			atomicAdd(resx + a, dx);
			atomicAdd(resy + a, dy);
		}
		__syncthreads();

		if (tix < 42)
		{
			re8x[tix] = resx[tix];
			re8y[tix] = resy[tix];
			for (int k = tix + 1; k < tix + 7; k++)
			{
				re8x[tix] += resx[k < 42 ? k : k - 42];
				re8y[tix] += resy[k < 42 ? k : k - 42];
			}
		}
		__syncthreads();

		if (tix == 0)
		{
			float maxr = 0.0f;
			int maxk = 0;
			for (int k = 0; k < 42; k++)
			{
				float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
				if (r > maxr)
				{
					maxr = r;
					maxk = k;
				}
			}
			float angle = dFastAtan2(re8y[maxk], re8x[maxk]);
			pt->angle = (angle < 0.0f ? angle + 2.0f * M_PI : angle);
		}
	}


	__global__ void gDescribe2(akaze::AkazePoint* points, void* tmem, int noctaves, int max_scale, int size2, int size3, int size4)
	{
#define EXTRACT_S 64
		__shared__ int acc_vals[3 * 30 * EXTRACT_S];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		int* acc_vals_im = &acc_vals[0];
		int* acc_vals_dx = &acc_vals[30 * EXTRACT_S];
		int* acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

		akaze::AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		float iratio = 1.f / (1 << o);
		int scale = (int)(pt->size + 0.5f);	// ?
		float xf = pt->x * iratio;
		float yf = pt->y * iratio;
		float ang = pt->angle;
		float co = __cosf(ang);
		float si = __sinf(ang);

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		int p = owhps[o].z;

		int* imd = (int*)tmem + offsets[o] + s * osizes[o];
		int* dxd = imd + max_scale * osizes[o] * 2;
		int* dyd = dxd + max_scale * osizes[o];
		int winsize = max(3 * size3, 4 * size4);

		for (int i = 0; i < 30; ++i)
		{
			int j = i * EXTRACT_S + tix;
			acc_vals_im[j] = 0;
			acc_vals_dx[j] = 0;
			acc_vals_dy[j] = 0;
		}
		__syncthreads();

		for (int i = tix; i < winsize * winsize; i += EXTRACT_S)
		{
			int y = i / winsize;
			int x = i - winsize * y;
			int m = max(x, y);
			if (m >= winsize)
				continue;
			int l = x - size2;
			int k = y - size2;
			int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
			int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
			int pos = yp * p + xp;
			int im = imd[pos];
			int dx = dxd[pos];
			int dy = dyd[pos];
			int rx = -dx * si + dy * co;
			int ry = dx * co + dy * si;

			if (m < 2 * size2)
			{
				int x2 = (x < size2 ? 0 : 1);
				int y2 = (y < size2 ? 0 : 1);
				// Add 2x2
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix] += im;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 3 * size3)
			{
				int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
				int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
				// Add 3x3
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 4 * size4)
			{
				int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
				int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
				// Add 4x4
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 2] += ry;
			}
		}
		__syncthreads();

		// Reduce stuff
		int acc_reg;
#pragma unroll
		for (int i = 0; i < 15; ++i)
		{
			// 0..31 takes care of even accs, 32..63 takes care of odd accs
			int offset = 2 * i + (tix < 32 ? 0 : 1);
			int tix_d = tix < 32 ? tix : tix - 32;
			for (int d = 0; d < 90; d += 30)
			{
				if (tix_d < 32)
				{
					acc_reg = acc_vals[3 * 30 * tix_d + offset + d] +
						acc_vals[3 * 30 * (tix_d + 32) + offset + d];
					acc_reg += shiftDown(acc_reg, 1);
					acc_reg += shiftDown(acc_reg, 2);
					acc_reg += shiftDown(acc_reg, 4);
					acc_reg += shiftDown(acc_reg, 8);
					acc_reg += shiftDown(acc_reg, 16);
				}
				if (tix_d == 0)
				{
					acc_vals[offset + d] = acc_reg;
				}
			}
		}

		__syncthreads();

		// Have 29*3 values to store
		// They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
		if (tix < 61)
		{
			unsigned char desc_r = 0;
#pragma unroll
			for (int i = 0; i < (tix == 60 ? 6 : 8); ++i)
			{
				int idx1 = comp_idx_1[tix * 8 + i];
				int idx2 = comp_idx_2[tix * 8 + i];
				desc_r |= (acc_vals[idx1] > acc_vals[idx2] ? 1 : 0) << i;
			}
			pt->features[tix] = desc_r;
		}
	}





	void createGaussKernel(float var, int radius)
	{
		static float _var = -1.f;
		static int _radius = 0;
		if (abs(_var - var) < 1e-3 && _radius == radius)
		{
			return;
		}

		_var = var;
		_radius = radius;

		const int ksz = radius + 1;
		std::unique_ptr<float> kptr(new float[ksz]);
		std::unique_ptr<int> ikptr(new int[ksz]);
		float denom = 1.f / (2.f * var);
		float* kernel = kptr.get();
		int* ikernel = ikptr.get();
		float ksum = 0;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] = expf(-i * i * denom);
			if (i == 0)
			{
				ksum += kernel[i];
			}
			else
			{
				ksum += kernel[i] + kernel[i];
			}
		}
		ksum = 1 / ksum;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] *= ksum;
			ikernel[i] = (int)(kernel[i] * 65536 + 0.5f);
		}

		CHECK(cudaMemcpyToSymbol(d_lowpass_kernel, ikernel, ksz * sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void hConv2dR2(unsigned char* src, int* dst, int width, int height, int pitch, float var)
	{
		createGaussKernel(var, 2);

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CheckMsg("hConv2dR2() execution failed\n");
	}


	void hConv2dR2(unsigned char* src, int* dst, int* temp, int width, int height, int pitch, float var)
	{
		createGaussKernel(var, 2);

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		//gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		gConv2dR2Row << <grid, block >> > (src, temp, width, height, pitch);
		gConv2dR2Col << <grid, block >> > (temp, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CheckMsg("hConv2dR2() execution failed\n");
	}


	void hConv2dR2(int* src, int* dst, int width, int height, int pitch, float var)
	{
		createGaussKernel(var, 2);

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CheckMsg("hConv2dR2() execution failed\n");
	}


	void hConv2dR2(int* src, int* dst, int* temp, int width, int height, int pitch, float var)
	{
		createGaussKernel(var, 2);

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		//gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		gConv2dR2Row << <grid, block >> > (src, temp, width, height, pitch);
		gConv2dR2Col << <grid, block >> > (temp, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CheckMsg("hConv2dR2() execution failed\n");
	}


	void hLowPass(unsigned char* src, int* dst, int width, int height, int pitch, float var, int ksz)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
		int r = 2;
#endif // LOG_TIME

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		if (ksz <= 5)
		{
#ifdef LOG_TIME
			r = 2;
#endif // LOG_TIME
			createGaussKernel(var, 2);
			gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 7)
		{
#ifdef LOG_TIME
			r = 3;
#endif // LOG_TIME
			createGaussKernel(var, 3);
			gConv2d<3> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 9)
		{
#ifdef LOG_TIME
			r = 4;
#endif // LOG_TIME
			createGaussKernel(var, 4);
			gConv2d<4> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 11)
		{
#ifdef LOG_TIME
			r = 5;
#endif // LOG_TIME
			createGaussKernel(var, 5);
			gConv2d<5> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else
		{
			std::cerr << "Kernels larger than 11 not implemented" << std::endl;
		}

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gConv2d<%d> %f\n", r, t0);
#endif // LOG_TIME
	}
	

	void hLowPass(unsigned char* src, int* dst, int* temp, int width, int height, int pitch, float var, int ksz)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
		int r = 2;
#endif // LOG_TIME

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		if (ksz <= 5)
		{
#ifdef LOG_TIME
			r = 2;
#endif // LOG_TIME
			createGaussKernel(var, 2);
			// gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
			gConv2dR2Row << <grid, block >> > (src, temp, width, height, pitch);
			gConv2dR2Col << <grid, block >> > (temp, dst, width, height, pitch);
		}
		else if (ksz <= 7)
		{
#ifdef LOG_TIME
			r = 3;
#endif // LOG_TIME
			createGaussKernel(var, 3);
			// gConv2d<3> << <grid, block >> > (src, dst, width, height, pitch);
			gConv2dRow<3> << <grid, block >> > (src, temp, width, height, pitch);
			gConv2dCol<3> << <grid, block >> > (temp, dst, width, height, pitch);
		}
		else if (ksz <= 9)
		{
#ifdef LOG_TIME
			r = 4;
#endif // LOG_TIME
			createGaussKernel(var, 4);
			// gConv2d<4> << <grid, block >> > (src, dst, width, height, pitch);
			gConv2dRow<4> << <grid, block >> > (src, temp, width, height, pitch);
			gConv2dCol<4> << <grid, block >> > (temp, dst, width, height, pitch);
		}
		else if (ksz <= 11)
		{
#ifdef LOG_TIME
			r = 5;
#endif // LOG_TIME
			createGaussKernel(var, 5);
			// gConv2d<5> << <grid, block >> > (src, dst, width, height, pitch);
			gConv2dRow<5> << <grid, block >> > (src, temp, width, height, pitch);
			gConv2dCol<5> << <grid, block >> > (temp, dst, width, height, pitch);
		}
		else
		{
			std::cerr << "Kernels larger than 11 not implemented" << std::endl;
		}

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gConv2d<%d> %f\n", r, t0);
#endif // LOG_TIME
	}


	void hDownWithSmooth(int* src, int* dst, int* smooth, int3 swhp, int3 dwhp)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		createGaussKernel(1.f, 2);
		dim3 block(X2, X2);
		dim3 grid((dwhp.x + X2 - 1) / X2, (dwhp.y + X2 - 1) / X2);
		gDownWithSmooth << <grid, block >> > (src, dst, smooth, swhp, dwhp);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gDownWithSmooth %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hDownWithSmooth() execution failed\n");
	}


	void hScharrContrast(int* src, int* grad, int& kcontrast, float per, int width, int height, int pitch)
	{
		// Compute gradient and find maximum gradient value
		int h_max_contrast = 1;
		int* d_max_contrast_addr = NULL;
		CHECK(cudaGetSymbolAddress((void**)&d_max_contrast_addr, d_max_contrast));
		CHECK(cudaMemcpy(d_max_contrast_addr, &h_max_contrast, sizeof(int), cudaMemcpyHostToDevice));

#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gScharrContrastWithMaximum << <grid, block >> > (src, grad, width, height, pitch);
		//gScharrContrastShared << <grid, block >> > (src, grad, width, height, pitch);
		gScharrContrastNaive << <grid, block >> > (src, grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
#endif // LOG_TIME

		//gFindMaxContrast << <grid, block >> > (grad, width, height, pitch);
		dim3 grid1((width / 2 + X2 - 1) / X2, (height / 2 + X2 - 1) / X2);
		gFindMaxContrastU4 << <grid1, block >> > (grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t1 = timer.read();
#endif // LOG_TIME

		CHECK(cudaMemcpy(&h_max_contrast, d_max_contrast_addr, sizeof(float), cudaMemcpyDeviceToHost));

		// Initialize Histogram
		int h_hist[NBINS];
		memset(h_hist, 0, NBINS * sizeof(int));
		CHECK(cudaMemcpyToSymbol(d_hist, h_hist, NBINS * sizeof(int), 0, cudaMemcpyHostToDevice));

		// Statistic histogram - constrast <= 4096
		int hfactor = (int)(NBINS / (float)h_max_contrast * 65536 + 0.5f);
		//float hfactor = (float)NBINS / (float)h_max_contrast;
		//gConstrastHist << <grid, block >> > (grad, hfactor, width, height, pitch);
		dim3 block2(32, 16);	// Must bigger than NBINS
		dim3 grid2((width + 32 - 1) / 32, (height + 16 - 1) / 16);
		gConstrastHistShared << <grid2, block2 >> > (grad, hfactor, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t2 = timer.read();
		printf("Time of gScharrContrastNaive %f\n", t0);
		printf("Time of gFindMaxContrastU4 %f\n", t1 - t0);
		printf("Time of gConstrastHist %f\n", t2 - t1);
#endif // LOG_TIME

		CHECK(cudaMemcpyFromSymbol(h_hist, d_hist, NBINS * sizeof(int), 0, cudaMemcpyDeviceToHost));

		// Compute contrast threshold
		int thresh = (width * height - h_hist[0]) * per;
		//printf("contrast threshold: %d\n", thresh);	// maybe a little change
		int cumuv = 0;
		int k = 1;
		while (k < NBINS)
		{
			if (cumuv >= thresh)
			{
				break;
			}
			cumuv += h_hist[k];
			k++;
		}
		kcontrast = k * h_max_contrast / NBINS;

		CheckMsg("hScharrContrast() execution failed\n");
	}


	void hHessianDeterminant(int* src, int* dx, int* dy, int step, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		float w = 10.f / 3.f;
		float fac1 = 1.f / (2.f * (w + 2.f));
		float fac2 = w * fac1;
		int ifac1 = (int)(fac1 * 65536 + 0.5f);
		int ifac2 = (int)(fac2 * 65536 + 0.5f);

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gDerivate << <grid, block >> > (src, dx, dy, step, ifac1, ifac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
#endif // LOG_TIME

		gHessianDeterminant << <grid, block >> > (dx, dy, src, step, ifac1, ifac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t1 = timer.read();
		printf("Time of gDerivate: %f\n", t0);
		printf("Time of gHessianDeterminant: %f\n", t1 - t0);
#endif // LOG_TIME

		CheckMsg("hHessianDeterminant() execution failed\n");
	}


	void hFlow(int* src, int* flow, akaze::DiffusivityType type, int kcontrast, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		float ikc = 1.f / (kcontrast * kcontrast);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gFlow << <grid, block >> > (src, flow, type, ikc, width, height, pitch);
		gFlowNaive << <grid, block >> > (src, flow, type, ikc, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gFlow: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hFlow() execution failed\n");
	}


	void hNldStep(int* img, int* flow, int* temp, float step_size, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		int stepfac = (int)(0.5f * step_size * 65536 + 0.5f);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		//gNldStep << <grid, block >> > (img, flow, temp, stepfac, width, height, pitch);
		gNldStepNaive << <grid, block >> > (img, flow, temp, stepfac, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gNldStep: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hNldStep() execution failed\n");
	}


	void hCalcExtremaMap(int* dets, int* response_map, float* size_map, int* layer_map, float* params,
		int octave, int max_scale, int threshold, int width, int height, int pitch, int opitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		CHECK(cudaMemcpyToSymbol(d_extrema_param, params, max_scale * 2 * sizeof(float), 0, cudaMemcpyHostToDevice));

		int psz = (int)params[0];	// The minimum border
		int depad = psz * 2;

		dim3 block(X2, X2);
		dim3 grid((width - depad + X2 - 1) / X2, (height - depad + X2 - 1) / X2, max_scale);
		gCalcExtremaMap << <grid, block >> > (dets, response_map, size_map, layer_map, octave, max_scale, psz,
			threshold, width, height, pitch, opitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gCalcExtremaMap: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hCalcExtremaMap() execution failed\n");
	}


	void hNmsR(akaze::AkazePoint* points, int* response_map, float* size_map, int* layer_map, int psz, int neigh, int width, int height, int pitch)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		int psz2 = psz + psz;
		dim3 block(X2, X2);
		dim3 grid((width - psz2 + X2 - 1) / X2, (height - psz2 + X2 - 1) / X2);
		int shared_radius = X2 + 2 * neigh;
		size_t shared_nbytes = shared_radius * shared_radius * sizeof(float);
		//gNmsR << <grid, block, shared_nbytes >> > (points, response_map, size_map, layer_map, psz, neigh, width, height, pitch);
		gNmsRNaive << <grid, block, shared_nbytes >> > (points, response_map, size_map, layer_map, psz, neigh, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gNmsR: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hNmsR() execution failed\n");
	}


	void hRefine(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		gRefine << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gRefine: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hRefine() execution failed\n");
	}


	void hCalcOrient(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		dim3 block(13 * 16);
		dim3 grid(result.num_pts);
		gCalcOrient << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of gCalcOrient: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hCalcOrient() execution failed\n");
	}


	void hDescribe(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale, int patsize)
	{
#ifdef LOG_TIME
		GpuTimer timer(0);
#endif // LOG_TIME

		int size2 = patsize;
		int size3 = ceilf(2.0f * patsize / 3.0f);
		int size4 = ceilf(0.5f * patsize);

		dim3 block(64);
		dim3 grid(result.num_pts);
		gDescribe2 << <grid, block >> > (result.d_data, tmem, noctaves, max_scale, size2, size3, size4);
		CHECK(cudaDeviceSynchronize());

#ifdef LOG_TIME
		float t0 = timer.read();
		printf("Time of descriptors computation: %f\n", t0);
#endif // LOG_TIME

		CheckMsg("hDescribe() execution failed\n");
	}

}