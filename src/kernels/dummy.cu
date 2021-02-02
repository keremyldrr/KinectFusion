#include "dummy.cuh"
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include "cuda_error_handle.h"
#include "Volume.h"

// #include <assert.h>

#include <stdio.h>
#define assert(X)                                                \
	if (!(X))                                                      \
		printf("tid %d: %s, %d\n", threadIdx.x, __FILE__, __LINE__); \
	return;

#define VOXSIZE 0.01f

#define MIN_DEPTH 0.2f
#define DISTANCE_THRESHOLD 2.f // inspired
#define MAX_WEIGHT_VALUE 128.f	 //inspired



__global__ void updateReconstructionKernel(
		Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize,
		cv::cuda::PtrStepSzf volume,
		CameraParameters cameraParams,
		cv::cuda::PtrStepSzf depthMap,
		Eigen::Matrix<float, 4, 4, Eigen::DontAlign> poseInverse)
{
	
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int lim = gridSize.x() *
					   gridSize.y() *
					   gridSize.z();
	
	//assert(gridSize.x() == 512);
	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> translation = poseInverse.block<3, 1>(0, 3);
	
	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation =poseInverse.block<3, 3>(0, 0);


	if (x >= 0 && x < gridSize.x() &&
			y >= 0 && y < gridSize.y())
	{
		for (auto z = 0; z < gridSize.z(); z++)
		{
			// TODO: Why now unsigned long long to avoid overflow?
			int  ind = (x * gridSize.y() + y) * gridSize.z() + z;
			if (ind < lim)
			{
				// printf("%d %d %d \n",x,y,z);
				int vx = x - ((gridSize.x() - 1) / 2);
				int vy = y - ((gridSize.y() - 1) / 2);
				int vz = z - ((gridSize.z() - 1) / 2);
				Vector3f voxelWorldPosition(vx + 0.5, vy + 0.5, vz + 0.5);
				voxelWorldPosition *= VOXSIZE; //TODO: //model.voxSize;

				Eigen::Matrix<float, 3, 1, Eigen::DontAlign> voxelCamPosition = rotation * voxelWorldPosition + translation;
				// voxelCamPosition = voxelCamPosition + translation;

				if (voxelCamPosition.z() < 0)
				{
					continue;
				}

				const Vector2i imagePosition(
						(voxelCamPosition.y() / voxelCamPosition.z()) * cameraParams.fovY + cameraParams.cY,
						(voxelCamPosition.x() / voxelCamPosition.z()) * cameraParams.fovX + cameraParams.cX);

				if (!(imagePosition.x() < 0 ||
							imagePosition.x() >= cameraParams.depthImageHeight ||
							imagePosition.y() < 0 ||
							imagePosition.y() >= cameraParams.depthImageWidth))
				{

					// const float depth = depthMap[imagePosition.x() * cameraParams.depthImageWidth + imagePosition.y()];
					const float depth = depthMap(imagePosition.x() ,imagePosition.y());
						if (depth > 0 )//&&  depth != MINF)
					{

						const Vector3f homogenImagePosition(
								(imagePosition.x() - cameraParams.cX) / cameraParams.fovX,
								(imagePosition.y() - cameraParams.cY) / cameraParams.fovY,
								1.0f);
						const float lambda = homogenImagePosition.norm();
						
						const float value = (-1.f) * ((1.f / lambda) * (voxelCamPosition).norm() - depth);
						if (value >= -DISTANCE_THRESHOLD)
						{
						
							const float sdfValue = fmin(1.f, value / DISTANCE_THRESHOLD);

							const float currValue = volume(ind, 0);
							const float currWeight = volume(ind, 1);

							const float addWeight = 1;
							const float nextTSDF =
									(currWeight * currValue + addWeight * sdfValue) /
									(currWeight + addWeight);
							// TODO: Check the MAX_WEIGHT_VALUE and how it would work after max iterations
							volume(ind, 0) = nextTSDF;
							volume(ind, 1) = fmin(currWeight + addWeight, MAX_WEIGHT_VALUE);
						}
					}
				}
			}
		}
	}
}

namespace Wrapper
{
	

	void updateReconstruction(Volume &model,
														const CameraParameters &cameraParams,
														const float *const depthMap,
														const MatrixXf &poseInverse)
	{
		std::vector<int> sizes{model.gridSize.x(),model.gridSize.y(),
													 model.gridSize.z()};
		cv::cuda::GpuMat deviceModel; //(sizes,CV_32FC2);
		cv::cuda::GpuMat anan;
		cv::Mat baban(3, 3, CV_32FC1);
		// TODO: Find better optimization for GPU Arch
		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(sizes[0] / threadsX, sizes[1] / threadsY);

	
		deviceModel.upload(model.getGrid());

		cv::Mat h_depthImage(cameraParams.depthImageHeight,cameraParams.depthImageWidth,CV_32FC1,(float *)depthMap);
		cv::cuda::GpuMat d_depthImage;
		d_depthImage.upload(h_depthImage);

		updateReconstructionKernel<<<blocks, threads>>>(
				model.gridSize,
				deviceModel,
				cameraParams,
				d_depthImage,
				poseInverse
				);

		cudaDeviceSynchronize();

		cv::Mat tempResult;


		deviceModel.download(tempResult);
		model.setGrid(tempResult);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			// Possibly: exit(-1) if program cannot continue....
		}



		cudaDeviceSynchronize();
	}
} // namespace Wrapper