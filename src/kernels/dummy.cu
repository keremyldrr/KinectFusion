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

// TODO: hardcoded in multiple places
#define MIN_DEPTH 0.2f
#define DISTANCE_THRESHOLD 2.f // inspired
#define MAX_WEIGHT_VALUE 128.f //inspired

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

	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation = poseInverse.block<3, 3>(0, 0);

	if (x >= 0 && x < gridSize.x() &&
			y >= 0 && y < gridSize.y())
	{
		for (auto z = 0; z < gridSize.z(); z++)
		{
			// TODO: Why now unsigned long long to avoid overflow?
			int ind = (x * gridSize.y() + y) * gridSize.z() + z;
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
					const float depth = depthMap(imagePosition.x(), imagePosition.y());
					if (depth > 0) //&&  depth != MINF)
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

__device__ bool isValid(Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize,
												Eigen::Matrix<float, 3, 1, Eigen::DontAlign> point)
{
	return point.x() < gridSize.x() / 2 && point.y() < gridSize.y() / 2 &&
				 point.z() < gridSize.z() / 2 && point.x() > -gridSize.x() / 2 &&
				 point.y() > -gridSize.y() / 2 && point.z() > -gridSize.z() / 2;
}

// @position should be in voxelCoordinates [-something, something]
__device__ float getFromVolume(cv::cuda::PtrStepSzf volume,
															 Eigen::Matrix<float, 3, 1, Eigen::DontAlign> position,
															 Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize)
{
	unsigned int vx = position.x() + ((gridSize.x() - 1) / 2);
	unsigned int vy = position.y() + ((gridSize.y() - 1) / 2);
	unsigned int vz = position.z() + ((gridSize.z() - 1) / 2);

	unsigned int ind = (vx * gridSize.y() + vy) * gridSize.z() + vz;
	// printf("%f -> ;) (%d, %d, %d)\n", volume(ind, 0), vx, vy, vz);
	return volume(ind, 0);
}

__device__ float interpolation(cv::cuda::PtrStepSzf volume,
															 Eigen::Matrix<float, 3, 1, Eigen::DontAlign> position,
															 Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize)
{
	Vector3f pointInGrid((int)position.x(), (int)position.y(), (int)position.z());
	return getFromVolume(volume, pointInGrid, gridSize);
}

// TODO: interpolation

__global__ void rayCastKernel(Eigen::Matrix<float, 4, 4, Eigen::DontAlign> cameraPose,
															CameraParameters params,
															Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize,
															cv::cuda::PtrStepSzf volume,
															cv::cuda::PtrStepSz<float3> surfacePoints,
															cv::cuda::PtrStepSz<float3> surfaceNormals)
{

	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (0 <= x && x <= params.depthImageWidth && 0 <= y && y <= params.depthImageHeight)
	{

		// surfacePoints(x, y) = make_float3(255.0, 255.0, 255.0);
		const Vector3f pixelInCameraCoords((x - params.cX) / params.fovX,
																			 (y - params.cY) / params.fovY, 1.0);

		Vector3f currPositionInCameraWorld = pixelInCameraCoords.normalized() * MIN_DEPTH;

		currPositionInCameraWorld += cameraPose.block<3, 1>(0, 3);
		Vector3f rayStepVec = pixelInCameraCoords.normalized() * VOXSIZE;
		// Rotate rayStepVec to 3D world
		rayStepVec = (cameraPose.block<3, 3>(0, 0) * rayStepVec);

		Vector3f voxelInGridCoords = currPositionInCameraWorld / VOXSIZE;
		Vector3f currPoint, currNormal;

		float currTSDF = 1.0;
		bool sign = true;
		bool prevSign = sign;

		int maxRayDist = 1000;

		while ((prevSign == sign) && isValid(gridSize, voxelInGridCoords))
		{
			currTSDF = getFromVolume(volume, voxelInGridCoords, gridSize);

			voxelInGridCoords = currPositionInCameraWorld / VOXSIZE;
			currPositionInCameraWorld += rayStepVec;

			prevSign = sign;
			sign = currTSDF >= 0;
		}
		// printf("OUT");
		if ((sign != prevSign) && isValid(gridSize, voxelInGridCoords))
		{
			// surfacePoint = currPositionInCameraWorld;
		}
		else
		{
			return;
		}

		Vector3f neighbor = voxelInGridCoords;
		neighbor.x() += 1;
		if (!isValid(gridSize, neighbor))
			return;
		const float Fx1 = interpolation(volume, neighbor, gridSize);

		neighbor = voxelInGridCoords;

		neighbor.x() -= 1;
		if (!isValid(gridSize, neighbor))
			return;
		const float Fx2 = interpolation(volume, neighbor, gridSize);

		currNormal.x() = (Fx1 - Fx2);

		neighbor = voxelInGridCoords;
		neighbor.y() += 1;
		if (!isValid(gridSize, neighbor))
			return;
		const float Fy1 = interpolation(volume, neighbor, gridSize);

		neighbor = voxelInGridCoords;
		neighbor.y() -= 1;
		if (!isValid(gridSize, neighbor))
			return;
		const float Fy2 = interpolation(volume, neighbor, gridSize);

		currNormal.y() = (Fy1 - Fy2);

		neighbor = voxelInGridCoords;
		neighbor.z() += 1;
		if (!isValid(gridSize, neighbor))
			return;
		const float Fz1 = interpolation(volume, neighbor, gridSize);

		neighbor = voxelInGridCoords;
		neighbor.z() -= 1;
		if (!isValid(gridSize, neighbor))
			return;
		const float Fz2 = interpolation(volume, neighbor, gridSize);

		currNormal.z() = (Fz1 - Fz2);

		if (currNormal.norm() == 0)
			return;

		currNormal.normalize();
		surfacePoints(y, x) = make_float3(currPoint.x(), currPoint.y(), currPoint.z());
		surfaceNormals(y, x) = make_float3(currNormal.x(), currNormal.y(), currNormal.z());

		// return true;
	}
	// TODO: set the value for surface point and normal
	// bool exists =
	// 		pointRay(cameraPose, params, y, x, currPoint, currNormal);
	// if (exists)
	// {
	// 	surfacePoints.push_back(currPoint);
	// 	surfaceNormals.push_back(currNormal);
	// }
}

namespace Wrapper
{
	void updateReconstruction(Volume &model,
														const CameraParameters &cameraParams,
														const float *const depthMap,
														const MatrixXf &poseInverse)
	{
		std::vector<int> sizes{model.gridSize.x(), model.gridSize.y(),
													 model.gridSize.z()};
		cv::cuda::GpuMat deviceModel; //(sizes,CV_32FC2);

		// TODO: Find better optimization for GPU Arch
		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(sizes[0] / threadsX, sizes[1] / threadsY);

		deviceModel.upload(model.getGrid());

		cv::Mat h_depthImage(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC1, (float *)depthMap);
		cv::cuda::GpuMat d_depthImage;
		d_depthImage.upload(h_depthImage);

		updateReconstructionKernel<<<blocks, threads>>>(
				model.gridSize,
				deviceModel,
				cameraParams,
				d_depthImage,
				poseInverse);

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
	}

	void rayCast(Volume &model,
							 const CameraParameters &cameraParams,
							 const MatrixXf &cameraPose)
	{
		std::cout << "raycasting" << std::endl;
		cv::Mat surfacePoints(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3),
				surfaceNormals(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		surfacePoints.setTo(0);
		surfaceNormals.setTo(0);

		cv::cuda::GpuMat deviceModel, deviceSurfacePoints, deviceSurfaceNormals; //(sizes,CV_32FC2);

		// TODO: Find better optimization for GPU Arch
		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(cameraParams.depthImageWidth / threadsX, cameraParams.depthImageHeight / threadsY);

		deviceModel.upload(model.getGrid());
		deviceSurfacePoints.upload(surfacePoints);
		deviceSurfaceNormals.upload(surfaceNormals);

		rayCastKernel<<<blocks, threads>>>(
				cameraPose,
				cameraParams,
				model.gridSize,
				deviceModel,
				deviceSurfacePoints,
				deviceSurfaceNormals);

		cudaDeviceSynchronize();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			// Possibly: exit(-1) if program cannot continue....
		}
		deviceSurfacePoints.download(surfacePoints);
		deviceSurfaceNormals.download(surfaceNormals);

		std::vector<Vector3f> points, normals;
		for (int i = 0; i < cameraParams.depthImageHeight; ++i)
		{
			for (int j = 0; j < cameraParams.depthImageWidth; ++j)
			{
				if (!(surfacePoints.at<cv::Vec3f>(i, j)[0] == 0 &&
							surfacePoints.at<cv::Vec3f>(i, j)[1] == 0 &&
							surfacePoints.at<cv::Vec3f>(i, j)[2] == 0))
				{
					points.push_back(Vector3f(surfacePoints.at<cv::Vec3f>(i, j)[0],
																		surfacePoints.at<cv::Vec3f>(i, j)[1],
																		surfacePoints.at<cv::Vec3f>(i, j)[2]));

					normals.push_back(Vector3f(surfaceNormals.at<cv::Vec3f>(i, j)[0],
																		 surfaceNormals.at<cv::Vec3f>(i, j)[1],
																		 surfaceNormals.at<cv::Vec3f>(i, j)[2]));
				}
			}
		}
		static int shitCounter = 0;

		cv::imwrite("DepthImage" + std::to_string(shitCounter++) + ".png", (surfaceNormals + 1.0f) / 2.0 *255.0f);
		PointCloud pcd(points, normals);
		model.setPointCloud(pcd);
	}
} // namespace Wrapper