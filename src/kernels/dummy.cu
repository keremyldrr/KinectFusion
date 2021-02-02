#include "dummy.cuh"
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include "cuda_error_handle.h"
#include "Volume.h"
#define BLOCK_ROWS 16
#define BLOCK_COLS 16



__global__ void test_kernel(cv::cuda::PtrStepSzf volume) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	// unsigned int index = row * (img.cols) + col;
	int dim = 512 // TODO: pass this as parameter;
	unsigned int lim = dim*dim*dim;
	if(x>=0 && x<dim && y>=0 && y<dim){
		for(auto z=0;z<dim;z++)
		{
			int ind = (x * dim + y) * dim + z;
			if(ind < lim){
			volume(ind,0) = x-y;
			volume(ind,1) = z;
			}
			// }
			// volume.ptr(x,y,z)[1] = 0;
			// volume.at<cv::Vec2f>(x,y,z)[0] = 0;
			// volume.at<cv::Vec2f>(x,y,z)[1] = 0;

		}
	}
	// *i = 99;

}

namespace Wrapper {
	void wrapper(cv::cuda::GpuMat &img, Volume &model)
	{
		std::vector<int> sizes;
		sizes.push_back(512);
		sizes.push_back(512);
		sizes.push_back(512);
		
		cv::cuda::GpuMat dummy;//(sizes,CV_32FC2);
		cv::Mat m;
		
	
		cv::Mat voxels = model.getGrid();

		//TODO THIS IS UGLY PLS OPTIMIZE
		const dim3 threads(32,32);
		const dim3 blocks(512/32,512/32);


		std::vector<int> flattenedSize;
		flattenedSize.push_back(model.gridSize.x()* model.gridSize.y() * model.gridSize.z());
		// flattenedSize.push_back(27);
		flattenedSize.push_back(1);
		
		
	

		dummy.upload(voxels.reshape(2,flattenedSize));


		cudaError_t err = cudaGetLastError();
		
		
		
		test_kernel <<<blocks,threads>>> (dummy);
		cudaDeviceSynchronize();
		dummy.download(m);
		m = m.reshape(2,sizes);
		model.setGrid(m);

		cudaError_t err = cudaGetLastError();

		if ( err != cudaSuccess )
		{
		   printf("CUDA Error: %s\n", cudaGetErrorString(err));       
   
		   // Possibly: exit(-1) if program cannot continue....
		}
        // d_img_ = d_img_original;
		cudaDeviceSynchronize();
	}
}