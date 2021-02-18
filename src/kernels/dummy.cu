#include "dummy.cuh"
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include "cuda_error_handle.h"
#include "Volume.h"
#include "Eigen.h"
#include <assert.h>
#include <stdio.h>

#define assert(X)                                                    \
	if (!(X))                                                        \
		printf("tid %d: %s, %d\n", threadIdx.x, __FILE__, __LINE__); \
	return;

#define ICP_DISTANCE_THRESHOLD 0.1f // inspired from excellence in m
// The angle threshold (as described in the paper) in degrees
#define ICP_ANGLE_THRESHOLD 15 // inspired from excellence in degrees
#define VOXSIZE 0.01
// TODO: hardcoded in multiple places
#define MIN_DEPTH 0.0f			 //in m
#define DISTANCE_THRESHOLD 0.02f //2.0f // inspired
#define MAX_WEIGHT_VALUE 128.f	 //inspired

__global__ void updateReconstructionKernel(
	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize,
	cv::cuda::PtrStepSzf volume,
	CameraParameters cameraParams,
	cv::cuda::PtrStepSzf depthMap,
	Eigen::Matrix<float, 4, 4, Eigen::DontAlign> modelToFrame,
	float minf)
{

	unsigned long long x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned long long y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned long long lim = gridSize.x() * gridSize.y() * gridSize.z();

	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> translation = modelToFrame.block<3, 1>(0, 3);
	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation = modelToFrame.block<3, 3>(0, 0);
	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> translationFrameToModel = -rotation * translation;

	if (x >= 0 && x < gridSize.x() &&
		y >= 0 && y < gridSize.y())
	{
		for (auto z = 0; z < gridSize.z(); z++)
		{
			unsigned long long ind = (x * gridSize.y() + y) * gridSize.z() + z;
			if (ind < lim)
			{
				int vx = x - ((gridSize.x() - 1) / 2);
				int vy = y - ((gridSize.y() - 1) / 2);
				int vz = z - ((gridSize.z() - 1) / 2);
				/* p */ Vector3f voxelWorldPosition = Vector3f(vx + 0.5, vy + 0.5, vz + 0.5) * VOXSIZE;

				Eigen::Matrix<float, 3, 1, Eigen::DontAlign> voxelCamPosition = rotation * voxelWorldPosition + translation;

				if (voxelCamPosition.z() < 0)
				{
					continue;
				}

				const Vector2i imagePosition(
					(voxelCamPosition.x() / voxelCamPosition.z()) * cameraParams.fovX + cameraParams.cX,
					(voxelCamPosition.y() / voxelCamPosition.z()) * cameraParams.fovY + cameraParams.cY);

				if (!(imagePosition.x() < 0 ||
					  imagePosition.x() >= cameraParams.depthImageWidth ||
					  imagePosition.y() < 0 ||
					  imagePosition.y() >= cameraParams.depthImageHeight))
				{
					const float depth = depthMap(imagePosition.y(), imagePosition.x());

					const float dv = 0.5f * (depthMap(imagePosition.y(), imagePosition.x() + 1) - depthMap(imagePosition.y(), imagePosition.x() - 1));
					const float du = 0.5f * (depthMap(imagePosition.y() + 1, imagePosition.x()) - depthMap(imagePosition.y() - 1, imagePosition.x()));
					if (depth > 0 && depth != minf && du != minf && dv != minf && !(abs(du) > 0.1f / 4 || abs(dv) > 0.1f / 4))
					{

						const Vector3f homogenImagePosition(
							(imagePosition.x() - cameraParams.cX) / cameraParams.fovX,
							(imagePosition.y() - cameraParams.cY) / cameraParams.fovY,
							1.0f);

						const float lambda = (homogenImagePosition).norm();

						// const float value = (-1.f) * ((1.f / lambda) * (voxelCamPosition).norm() - depth);
						// const float value = (translationFrameToModel - voxelWorldPosition).norm() - depth;

						// const float sdfValue = (value > 0) ? fmin(1.f, value / DISTANCE_THRESHOLD) : fmax(-1.f, value / DISTANCE_THRESHOLD);
						const float value = (-1.f) * ((1.f / lambda) * (translationFrameToModel - voxelWorldPosition).norm() - depth);
						if (value >= -DISTANCE_THRESHOLD)
						{

							const float sdfValue = fmin(1.f, value / DISTANCE_THRESHOLD);

							const float currValue = volume(ind, 0);
							const float currWeight = volume(ind, 1);

							const float addWeight = 1;
							const float nextTSDF =
								(currWeight * currValue + addWeight * sdfValue) /
								(currWeight + addWeight);
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
	unsigned long long vx = position.x() + ((gridSize.x() - 1) / 2);
	unsigned long long vy = position.y() + ((gridSize.y() - 1) / 2);
	unsigned long long vz = position.z() + ((gridSize.z() - 1) / 2);

	unsigned long long ind = (vx * gridSize.y() + vy) * gridSize.z() + vz;
	// printf("%f -> ;) (%d, %d, %d)\n", volume(ind, 0), vx, vy, vz);
	return volume(ind, 0);
}

__device__ float interpolation(cv::cuda::PtrStepSzf volume,
							   Eigen::Matrix<float, 3, 1, Eigen::DontAlign> position,
							   Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize)
{
	Vector3f pointInGrid((int)position.x(), (int)position.y(), (int)position.z());

	// Toggle to disable interpolation
	//}	return getFromVolume(volume, pointInGrid, gridSize);

	Vector3f voxelCenter(pointInGrid.x() + 0.5f, pointInGrid.y() + 0.5f,
						 pointInGrid.z() + 0.5f);

	pointInGrid.x() = (position.x() < voxelCenter.x()) ? (pointInGrid.x() - 1)
													   : pointInGrid.x();
	pointInGrid.y() = (position.y() < voxelCenter.y()) ? (pointInGrid.y() - 1)
													   : pointInGrid.y();
	pointInGrid.z() = (position.z() < voxelCenter.z()) ? (pointInGrid.z() - 1)
													   : pointInGrid.z();

	// pointInGrid = Vector3f(pointInGrid.x() - 1, pointInGrid.y() - 1,
	// pointInGrid.z() - 1);

	// Check Distance correctness
	const float distX = abs((abs(position.x()) - abs((pointInGrid.x()) + 0.5f)));
	const float distY = abs((abs(position.y()) - abs((pointInGrid.y()) + 0.5f)));
	const float distZ = abs((abs(position.z()) - abs((pointInGrid.z()) + 0.5f)));

	// TODO: Check the correctness of below, just a sanity check
	return (isValid(gridSize, pointInGrid)
				? getFromVolume(volume, pointInGrid, gridSize)
				: 0.0f) *
			   (1 - distX) * (1 - distY) * (1 - distZ) +
		   (isValid(gridSize,
					Vector3f(pointInGrid.x(), pointInGrid.y(), pointInGrid.z() + 1))
				? getFromVolume(volume, Vector3f(pointInGrid.x(), pointInGrid.y(), pointInGrid.z() + 1), gridSize)
				: 0.0f) *
			   (1 - distX) * (1 - distY) * (distZ) +
		   (isValid(gridSize,
					Vector3f(pointInGrid.x(), pointInGrid.y() + 1, pointInGrid.z()))
				? getFromVolume(volume, Vector3f(pointInGrid.x(), pointInGrid.y() + 1, pointInGrid.z()), gridSize)
				: 0.0f) *
			   (1 - distX) * distY * (1 - distZ) +
		   (isValid(gridSize, Vector3f(pointInGrid.x(), pointInGrid.y() + 1,
									   pointInGrid.z() + 1))
				? getFromVolume(volume, Vector3f(pointInGrid.x(), pointInGrid.y() + 1, pointInGrid.z() + 1), gridSize)
				: 0.0f) *
			   (1 - distX) * distY * distZ +
		   (isValid(gridSize,
					Vector3f(pointInGrid.x() + 1, pointInGrid.y(), pointInGrid.z()))
				? getFromVolume(volume, Vector3f(pointInGrid.x() + 1, pointInGrid.y(), pointInGrid.z()), gridSize)
				: 0.0f) *
			   distX * (1 - distY) * (1 - distZ) +
		   (isValid(gridSize, Vector3f(pointInGrid.x() + 1, pointInGrid.y(),
									   pointInGrid.z() + 1))
				? getFromVolume(volume, Vector3f(pointInGrid.x() + 1, pointInGrid.y(), pointInGrid.z() + 1), gridSize)
				: 0.0f) *
			   distX * (1 - distY) * distZ +
		   (isValid(gridSize, Vector3f(pointInGrid.x() + 1, pointInGrid.y() + 1,
									   pointInGrid.z()))
				? getFromVolume(volume, Vector3f(pointInGrid.x() + 1, pointInGrid.y() + 1, pointInGrid.z()), gridSize)
				: 0.0f) *
			   distX * distY * (1 - distZ) +
		   (isValid(gridSize, Vector3f(pointInGrid.x() + 1, pointInGrid.y() + 1,
									   pointInGrid.z() + 1))
				? getFromVolume(volume, Vector3f(pointInGrid.x() + 1, pointInGrid.y() + 1, pointInGrid.z() + 1), gridSize)
				: 0.0f) *
			   distX * distY * distZ;
}

// TODO: interpolation

__global__ void rayCastKernel(Eigen::Matrix<float, 4, 4, Eigen::DontAlign> frameToModel,
							  CameraParameters params,
							  Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize,
							  cv::cuda::PtrStepSzf volume,
							  cv::cuda::PtrStepSz<float3> surfacePoints,
							  cv::cuda::PtrStepSz<float3> surfaceNormals)
{

	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < params.depthImageWidth &&
		y < params.depthImageHeight)
	{

		Matrix3f intrinsicsInverse;
		intrinsicsInverse << 1/params.fovX, 0 , -params.cX/params.fovX,
							0, 1/params.fovY, -params.cY/params.fovY,
							0,0,1;
		Vector3f rayNext((x),(y) , 1.0f);

		Vector3f rayStart(0.f,0.f,0.f);

		rayStart += frameToModel.block<3, 1>(0, 3) / VOXSIZE;
		// Vector3f rayStepVec = pixelInCameraCoords.normalized() * VOXSIZE;
		rayNext = intrinsicsInverse * rayNext;

		// Rotate rayStepVec to 3D world
		rayNext = (frameToModel.block<3, 3>(0, 0) * rayNext + frameToModel.block<3, 1>(0, 3));
		rayNext /= VOXSIZE;

		Vector3f rayDir = (rayNext - rayStart).normalized() * VOXSIZE/20.0f;
		if(rayDir == Vector3f{ 0.0f, 0.0f, 0.0f })
			return;
		Vector3f currPositionInCameraWorld = rayStart * VOXSIZE;
		Vector3f voxelInGridCoords = currPositionInCameraWorld / VOXSIZE;
		Vector3f currPoint, currNormal;

		float currTSDF = getFromVolume(volume, rayStart, gridSize);
		bool sign = true;
		bool prevSign = sign;

		float prevTSDF = currTSDF;
		while ((prevSign == sign) && isValid(gridSize, voxelInGridCoords))
		{
			prevTSDF = currTSDF;
			currTSDF = getFromVolume(volume, voxelInGridCoords, gridSize);

			voxelInGridCoords = currPositionInCameraWorld / VOXSIZE;
			currPositionInCameraWorld += rayDir;

			prevSign = sign;
			sign = currTSDF >= 0;
		}

		if ((sign != prevSign) && isValid(gridSize, voxelInGridCoords))
		{
			// currPoint = currPositionInCameraWorld - rayStepVec * prevTSDF / (currTSDF - prevTSDF);
			// currPositionInCameraWorld += rayStepVec;
			currPoint = currPositionInCameraWorld;
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

		// if (currNormal.x() * currNormal.y() * currNormal.z() == 0)
		// 	return;

		// if (abs(currNormal.x()) > 0.1f || abs(currNormal.y()) > 0.1f || abs(currNormal.z()) > 0.1f)
		// 	return;

		currNormal.normalize();

		surfacePoints(y, x) = make_float3(currPoint.x(), currPoint.y(), currPoint.z());
		surfaceNormals(y, x) = make_float3(currNormal.x(), currNormal.y(), currNormal.z());
	}
}

__global__ void findCorrespondencesKernel(Eigen::Matrix<float, 4, 4, Eigen::DontAlign> modelToFrame,
										  Eigen::Matrix<float, 4, 4, Eigen::DontAlign> estimatedCameraPose,
										  CameraParameters cameraParams,
										  cv::cuda::PtrStepSz<float3> surfacePoints,
										  cv::cuda::PtrStepSz<float3> surfaceNormals,
										  cv::cuda::PtrStepSz<float3> sourceVertexMap,
										  cv::cuda::PtrStepSz<float3> sourceNormalMap,
										  cv::cuda::PtrStepSz<int2> matches,
										  float minf)
{

	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	Eigen::Matrix<float, 4, 4, Eigen::DontAlign> estimatedFrameToFrame = modelToFrame * estimatedCameraPose;

	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> estimatedFrametoFrameTranslation = estimatedFrameToFrame.block<3, 1>(0, 3);
	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> estimatedFrameToFrameRotation = estimatedFrameToFrame.block<3, 3>(0, 0);

	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> estimatedFrameToModelTranslation = estimatedCameraPose.block<3, 1>(0, 3);
	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> estimatedFrameToModelRotation = estimatedCameraPose.block<3, 3>(0, 0);

	if (x < cameraParams.depthImageWidth && y < cameraParams.depthImageHeight)
	{

		Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceNormal;
		sourceNormal.x() = sourceNormalMap(y, x).x;
		sourceNormal.y() = sourceNormalMap(y, x).y;
		sourceNormal.z() = sourceNormalMap(y, x).z;

		if (!(sourceNormal.x() == 0 &&
			  sourceNormal.y() == 0 &&
			  sourceNormal.z() == 0) &&
			!(sourceNormal.x() == minf &&
			  sourceNormal.y() == minf &&
			  sourceNormal.z() == minf))
			{

				Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertex;
				sourceVertex.x() = sourceVertexMap(y, x).x;
				sourceVertex.y() = sourceVertexMap(y, x).y;
				sourceVertex.z() = sourceVertexMap(y, x).z;

				Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertexPrevCamera = estimatedFrameToFrameRotation * sourceVertex + estimatedFrametoFrameTranslation;
				Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertexGlobal = estimatedFrameToModelRotation * sourceVertex + estimatedFrameToModelTranslation;
				// Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertexPrevCamera = sourceVertex ;
				// Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertexGlobal =  sourceVertex ;

				Eigen::Vector2f prevPixel;
			prevPixel.x() = __float2int_rd(sourceVertexPrevCamera.x() * cameraParams.fovX / sourceVertexPrevCamera.z() + cameraParams.cX + 0.5f);
			prevPixel.y() = __float2int_rd(sourceVertexPrevCamera.y() * cameraParams.fovY / sourceVertexPrevCamera.z() + cameraParams.cY + 0.5f);
				if (prevPixel.x() >= 0 && prevPixel.y() >= 0 &&
					prevPixel.x() < cameraParams.depthImageWidth &&
					prevPixel.y() < cameraParams.depthImageHeight &&
					sourceVertexPrevCamera.z() > 0)
				{


					Eigen::Matrix<float, 3, 1, Eigen::DontAlign> oldNormal;

					float bestCos = ICP_ANGLE_THRESHOLD;
					for (int offX = 0; offX <= 0; offX++)
					{
						for (int offY = 0; offY <= 0; offY++)
						{
							oldNormal.x() = surfaceNormals(prevPixel.y() + offY, prevPixel.x() + offX).x;
							oldNormal.y() = surfaceNormals(prevPixel.y() + offY, prevPixel.x() + offX).y;
							oldNormal.z() = surfaceNormals(prevPixel.y() + offY, prevPixel.x() + offX).z;

							if (!(oldNormal.x() == 0 &&
								  oldNormal.y() == 0 &&
								  oldNormal.z() == 0) &&
								!(oldNormal.x() == minf &&
								  oldNormal.y() == minf &&
								  oldNormal.z() == minf))
							{

								Eigen::Matrix<float, 3, 1, Eigen::DontAlign> oldVertex;

								oldVertex.x() = surfacePoints(prevPixel.y() + offY, prevPixel.x() + offX).x;
								oldVertex.y() = surfacePoints(prevPixel.y() + offY, prevPixel.x() + offX).y;
								oldVertex.z() = surfacePoints(prevPixel.y() + offY, prevPixel.x() + offX).z;
								const float distance = (oldVertex - sourceVertexGlobal).norm();
								if (distance <= ICP_DISTANCE_THRESHOLD)
								{

									Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceNormalGlobal = (estimatedFrameToModelRotation * sourceNormal);
									const float cos = (sourceNormalGlobal.dot(oldNormal));
								// const float cos = acos(sourceNormalGlobal.dot(oldNormal)) * 180 / EIGEN_PI;

									if (abs(cos) >= 0.5 &&  abs(cos) <= 1.1f)
								// if (cos < bestCos)
									{
										matches(y, x) = make_int2(prevPixel.y() + offY, prevPixel.x() + offX);
										// matches(y, x) = make_int2(y, x);
										// bestCos = cos;
									}
									else
									{
									
									// printf("%d %d %f %f %f %f %f\n ",x,y,prevPixel.x(),prevPixel.y(),sourceVertexPrevCamera.x(),sourceVertexPrevCamera.y(),sourceVertexPrevCamera.z());

									//	printf("INVALID ANGLE : %f %f %f %f   \n",cos, oldNormal.x()-sourceNormalGlobal.x(),oldNormal.y()-sourceNormalGlobal.y(),oldNormal.z() - sourceNormalGlobal.z());
									}
								}
								else
								{

									//printf("INVALID DISTANCE : %f \n", distance);
								}
							}
						}
					}
				}
			}
	}
}

PointCloud depthNormalMapToPcd(const cv::Mat &vertexMap, const cv::Mat &normalMap)
{

	std::vector<Vector3f> vertices;
	std::vector<Vector3f> normals;
	for (int i = 0; i < vertexMap.rows; i++)
	{

		for (int j = 0; j < vertexMap.cols; j++)
		{
			if (vertexMap.at<cv::Vec3f>(i, j)[0] != MINF && normalMap.at<cv::Vec3f>(i, j)[0] != MINF

			)
			{
				if (((!(vertexMap.at<cv::Vec3f>(i, j)[0] == 0 &&
						vertexMap.at<cv::Vec3f>(i, j)[1] == 0 &&
						vertexMap.at<cv::Vec3f>(i, j)[2] == 0))) &&
					((!(normalMap.at<cv::Vec3f>(i, j)[0] == 0 &&
						normalMap.at<cv::Vec3f>(i, j)[1] == 0 &&
						normalMap.at<cv::Vec3f>(i, j)[2] == 0))))
				{
					Vector3f vert(vertexMap.at<cv::Vec3f>(i, j)[0], vertexMap.at<cv::Vec3f>(i, j)[1], vertexMap.at<cv::Vec3f>(i, j)[2]);
					Vector3f normal(normalMap.at<cv::Vec3f>(i, j)[0], normalMap.at<cv::Vec3f>(i, j)[1], normalMap.at<cv::Vec3f>(i, j)[2]);
					vertices.push_back(vert);
					normals.push_back(normal);
				}
			}

			else
			{
				// std::cout << "MINF" << std::endl;
			}
		}
	}

	return PointCloud(vertices, normals);
}
namespace Wrapper
{
	void updateReconstruction(Volume &model,
							  const CameraParameters &cameraParams,
							  const float *const depthMap,
							  const MatrixXf &modelToFrame)
	{
		std::vector<int> sizes{model.gridSize.x(), model.gridSize.y(),
							   model.gridSize.z()};
		cv::cuda::GpuMat deviceModel; //(sizes,CV_32FC2);

		// TODO: Find better optimization for GPU Arch
		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(sizes[0] / threadsX, sizes[1] / threadsY);

		cv::Mat h_depthImage(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC1, (float *)depthMap);
		cv::cuda::GpuMat d_depthImage;
		d_depthImage.upload(h_depthImage);
		updateReconstructionKernel<<<blocks, threads>>>(
			model.gridSize,
			model.getGPUGrid(),
			cameraParams,
			d_depthImage,
			modelToFrame,
			MINF);

		cudaDeviceSynchronize();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			// Possibly: exit(-1) if program cannot continue....
		}
	}

	void rayCast(Volume &model,
				 CameraParameters cameraParams,
				 const MatrixXf &frameToModel, int level)
	{
		// TODO: Find better optimization for GPU Arch
		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(cameraParams.depthImageWidth / threadsX, cameraParams.depthImageHeight / threadsY);

		float scaleFactor = pow(0.5, level);

		cameraParams.fovX *= scaleFactor;
		cameraParams.fovY *= scaleFactor;
		cameraParams.cX *= scaleFactor;
		cameraParams.cY *= scaleFactor;
		cameraParams.depthImageHeight *= scaleFactor;
		cameraParams.depthImageWidth *= scaleFactor;

		cv::Mat surfacePoints(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		cv::Mat surfaceNormals(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		surfacePoints.setTo(0);
		surfaceNormals.setTo(0);

		cv::cuda::GpuMat deviceModel, deviceSurfacePoints, deviceSurfaceNormals; //(sizes,CV_32FC2);
		deviceSurfacePoints.upload(surfacePoints);
		deviceSurfaceNormals.upload(surfaceNormals);

		rayCastKernel<<<blocks, threads>>>(
			frameToModel,
			cameraParams,
			model.gridSize,
			model.getGPUGrid(),
			deviceSurfacePoints,
			deviceSurfaceNormals);

		cudaDeviceSynchronize();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}
		deviceSurfacePoints.download(surfacePoints);
		deviceSurfaceNormals.download(surfaceNormals);

		static int imageCounter = 0;
		// if(imageCounter == 0){
		model.setSurfaceNormals(surfaceNormals, level);
		model.setSurfacePoints(surfacePoints, level);
		// }
		if (level == 0)
		{
			if (imageCounter >= 0)
			{
				cv::imwrite("DepthImage" + std::to_string(imageCounter) + ".png", (surfaceNormals + 1.0f) / 2.0 * 255.0f);

				PointCloud pcd = depthNormalMapToPcd(surfacePoints, surfaceNormals);
				// model.setPointCloud(pcd);
				// pcd.writeMesh("predictedSurface" + std::to_string(imageCounter) + "_Level_" + std::to_string(level) + ".off");
			}
			imageCounter++;
		}
	}

	bool poseEstimation(VirtualSensor &sensor,
						Matrix4f &frameToModel,
						CameraParameters cameraParams,
						cv::cuda::GpuMat &surfacePoints,
						cv::cuda::GpuMat &surfaceNormals,
						int level)
	{

		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(cameraParams.depthImageWidth / threadsX,
						  cameraParams.depthImageHeight / threadsY);

		// int iters[3]{10, 5, 3};
		int iters[3]{10, 5, 3};

		float scaleFactor = pow(0.5, level);
		cameraParams.fovX *= scaleFactor;
		cameraParams.fovY *= scaleFactor;
		cameraParams.cX *= scaleFactor;
		cameraParams.cY *= scaleFactor;
		cameraParams.depthImageHeight *= scaleFactor;
		cameraParams.depthImageWidth *= scaleFactor;

		cv::cuda::GpuMat sourceVertexMap;
		cv::Mat hostSourceVertexMap(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		hostSourceVertexMap.setTo(0);
		hostSourceVertexMap = sensor.getVertexMap(level);
		cv::cuda::GpuMat sourceNormalMap;
		cv::Mat hostSourceNormalMap(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		hostSourceNormalMap.setTo(0);
		hostSourceNormalMap = sensor.getNormalMap(level);

		cv::Mat targetPointsMat(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		targetPointsMat.setTo(0);
		
		cv::Mat targetNormalsMat(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		targetNormalsMat.setTo(0);
		
		sourceVertexMap.upload(hostSourceVertexMap);
		sourceNormalMap.upload(hostSourceNormalMap);

		cv::cuda::GpuMat matches;
		cv::Mat hostMatches(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32SC2);
		matches.setTo(0);
		hostMatches.setTo(0);

		cv::Mat hostDistances(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC1);
		hostDistances.setTo(0);
		cv::cuda::GpuMat deviceDistances;
		deviceDistances.upload(hostDistances);

		Matrix4f estimatedCameraPose = frameToModel;	//initial
		Matrix4f modelToFrame = frameToModel.inverse(); //previous frame to model
		cv::Mat image1;

		surfacePoints.download(targetPointsMat);
		surfaceNormals.download(targetNormalsMat);
		
		cv::hconcat((targetNormalsMat + 1) * 255 / 2.0f, (hostSourceNormalMap + 1) * 255 / 2.0f, image1); //Syntax-> hconcat(source1,source2,destination);
		static int q = 0;
		int minfCount = 0;

		for (int iter = 0; iter < iters[level]; iter++)
		{
			hostMatches.setTo(0);
			matches.upload(hostMatches);

			hostDistances.setTo(0);
			deviceDistances.upload(hostDistances);
			cudaDeviceSynchronize();

			// TODO: replace groundTruth by estimatedCameraPose
			findCorrespondencesKernel<<<blocks, threads>>>(
				modelToFrame, estimatedCameraPose, cameraParams, surfacePoints,
				surfaceNormals, sourceVertexMap, sourceNormalMap, matches, MINF);
			cudaDeviceSynchronize();
			

			cudaError_t err = cudaGetLastError();
			matches.download(hostMatches);
			cv::Mat splittedMatches[3];
			cv::split(hostMatches, splittedMatches);

			int nzCount = cv::countNonZero(splittedMatches[0]);
			std::cout << nzCount << std::endl;

			std::vector<Vector3f> sourcePts;
			std::vector<Vector3f> targetPts;
			std::vector<Vector3f> targetNormals;

			Matrix3f frameToModelRotation = estimatedCameraPose.block<3, 3>(0, 0);
			Vector3f frameToModelTranslation = estimatedCameraPose.block<3, 1>(0, 3);

			// Matrix3f gtRotation = groundTruth.block<3, 3>(0, 0);
			// Vector3f gtTranslation = groundTruth.block<3, 1>(0, 3);
			int checkyboi = 0;
			minfCount = 0;
			for (int i = 0; i < cameraParams.depthImageHeight; i++)
			{
				for (int j = 0; j < cameraParams.depthImageWidth; j++)
				{
					checkyboi++;
					if (targetNormalsMat.at<cv::Vec3f>(i, j)[0] == MINF || targetNormalsMat.at<cv::Vec3f>(i, j)[0] == 0)
						minfCount++;
					// if (hostSourceVertexMap.at<cv::Vec3f>(i, j)[0] != MINF && hostSourceNormalMap.at<cv::Vec3f>(i, j)[0] != MINF)
					if (hostMatches.at<cv::Vec2i>(i, j)[0] != 0 && hostMatches.at<cv::Vec2i>(i, j)[1] != 0)
					{

						int targetX = hostMatches.at<cv::Vec2i>(i, j)[0];
						int targetY = hostMatches.at<cv::Vec2i>(i, j)[1];
						cv::Point src(j, i);
						cv::Point trgt(targetY + targetNormalsMat.cols, targetX);
						if (checkyboi % 5000 == 0)
						{
							int thickness = 1;
							int lineType = cv::LINE_8;
							cv::line(image1,
									 src,
									 trgt,
									 cv::Scalar(0, 0, 255),
									 thickness,
									 lineType);
						}
						Vector3f sourcepoint(
							hostSourceVertexMap.at<cv::Vec3f>(i, j)[0],
							hostSourceVertexMap.at<cv::Vec3f>(i, j)[1],
							hostSourceVertexMap.at<cv::Vec3f>(i, j)[2]);

						Vector3f targetPoint(
							targetPointsMat.at<cv::Vec3f>(targetX, targetY)[0],
							targetPointsMat.at<cv::Vec3f>(targetX, targetY)[1],
							targetPointsMat.at<cv::Vec3f>(targetX, targetY)[2]);
						Vector3f targetNormal(
							targetNormalsMat.at<cv::Vec3f>(targetX, targetY)[0],
							targetNormalsMat.at<cv::Vec3f>(targetX, targetY)[1],
							targetNormalsMat.at<cv::Vec3f>(targetX, targetY)[2]);
						// Vector3f targetNormal(
						// 	hostSourceNormalMap.at<cv::Vec3f>(i, j)[0],
						// 	hostSourceNormalMap.at<cv::Vec3f>(i, j)[1],
						// 	hostSourceNormalMap.at<cv::Vec3f>(i, j)[2]);

						sourcePts.push_back(frameToModelRotation * sourcepoint + frameToModelTranslation);
						targetNormals.push_back(targetNormal);
						targetPts.push_back(targetPoint);
						// targetNormals.push_back(gtRotation * targetNormal);
						// targetPts.push_back(gtRotation * sourcepoint + gtTranslation);
					}
				}
			}
			// cv::imwrite(std::to_string(q) + "merged" + std::to_string(iter) + ".png", image1);
			if (err != cudaSuccess)
			{
				printf("CUDA Error: %s\n", cudaGetErrorString(err));
			}
			if (sourcePts.size() == 0)
			{
				std::cout << "NO PONTS \n";
				return false;
			}

			auto increment = estimatePosePointToPlaneBefore(sourcePts, targetPts, targetNormals);
			estimatedCameraPose = increment * estimatedCameraPose;
		}

		std::cout << "MINF COUNT IN RAYCAST " << minfCount << std::endl;
		q++;

		// std::cout << frameToModel << std::endl;
		// std::cout << groundTruth << std::endl;
		std::cout << "***************" << std::endl;
		// std::cout << estimatedCameraPose - groundTruth << std::endl;
		frameToModel = estimatedCameraPose;
		return true;
	}

	Matrix4f estimatePosePointToPlaneBefore(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals)
	{
		const unsigned nPoints = sourcePoints.size();
		// Build the system
		MatrixXf A = MatrixXf::Zero(1 * nPoints, 6);
		VectorXf b = VectorXf::Zero(1 * nPoints);

		for (unsigned i = 0; i < nPoints; i++)
		{
			const auto &s = sourcePoints[i];
			const auto &d = targetPoints[i];
			const auto &n = targetNormals[i];

			// TODO: [DONE] Add the point-to-plane constraints to the system

			//  1 point-to-plane row per point
			A(i, 1) = n.x() * s.z() - n.z() * s.x();
			A(i, 0) = n.z() * s.y() - n.y() * s.z();
			A(i, 2) = n.y() * s.x() - n.x() * s.y();
			A(i, 3) = n.x();
			A(i, 4) = n.y();
			A(i, 5) = n.z();
			b(i) = n.x() * d.x() + n.y() * d.y() + n.z() * d.z() - n.x() * s.x() - n.y() * s.y() - n.z() * s.z();
		}

		VectorXf x(6);

		JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
		const MatrixXf &E_i = svd.singularValues().asDiagonal().inverse();
		const MatrixXf &U_t = svd.matrixU().transpose();
		const MatrixXf &V = svd.matrixV();

		x = V * E_i * U_t * b;
		// x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
		float alpha = x(0), beta = x(1), gamma = x(2);

		Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
							AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
							AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

		Vector3f translation = x.tail(3);

		Matrix4f estimatedPose = Matrix4f::Identity();

		estimatedPose.block<3, 3>(0, 0) = rotation;
		estimatedPose.block<3, 1>(0, 3) = translation;

		return estimatedPose;
	}
	Matrix4f estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals)
	{
		const unsigned nPoints = sourcePoints.size();
		// Build the system
		MatrixXf At = MatrixXf::Zero(6, 6);
		VectorXf b = VectorXf::Zero(6);
		MatrixXf AtA = MatrixXf::Zero(6, 6);
		VectorXf AtB = VectorXf::Zero(6);
		MatrixXf G = MatrixXf::Zero(3, 6);

		for (unsigned i = 0; i < nPoints; i++)
		{
			const auto &s = sourcePoints[i];
			const auto &d = targetPoints[i];
			const auto &n = targetNormals[i];

			Eigen::Matrix3f s_hat;
			s_hat << 0, -s(2), s(1),
				s(2), 0, -s(0),
				-s(1), s(0), 0;

			G.block<3, 3>(0, 0) = s_hat;
			G.block<3, 3>(0, 3) = Matrix3f::Identity();
			// TODO: [DONE] Add the point-to-plane constraints to the system
			At = G.transpose() * n;
			b = n.transpose() * (d - s);

			AtA += At * At.transpose();
			AtB += At * b;
		}

		Eigen::Matrix<float, 6, 1> x{AtA.fullPivLu().solve(AtB).cast<float>()};

		float alpha = x(2);
		float beta = x(0);
		float gamma = x(1);
		Matrix3f rotation;
		rotation << 1, alpha, -gamma,
			-alpha, 1, beta,
			gamma, -beta, 1;

		Vector3f translation = x.tail(3);
		Matrix4f poseIncrement = Matrix4f::Identity();

		poseIncrement.block<3, 3>(0, 0) = rotation;
		poseIncrement.block<3, 1>(0, 3) = translation;
		return poseIncrement;
	}
} // namespace Wrapper