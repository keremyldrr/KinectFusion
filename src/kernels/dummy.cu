#include "dummy.cuh"
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include "cuda_error_handle.h"
#include "Volume.h"
#include <assert.h>
#include <stdio.h>

#define assert(X)                                                    \
	if (!(X))                                                        \
		printf("tid %d: %s, %d\n", threadIdx.x, __FILE__, __LINE__); \
	return;

#define ICP_DISTANCE_THRESHOLD 0.1f // inspired from excellence in m
// The angle threshold (as described in the paper) in degrees
#define ICP_ANGLE_THRESHOLD 20 // inspired from excellence in degrees
#define VOXSIZE 0.01f		   // in m
// TODO: hardcoded in multiple places
#define MIN_DEPTH 0.2f		   //in m
#define DISTANCE_THRESHOLD 2.f // inspired
#define MAX_WEIGHT_VALUE 128.f //inspired
#define ICP_ITERATIONS 10
__global__ void updateReconstructionKernel(
	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> gridSize,
	cv::cuda::PtrStepSzf volume,
	CameraParameters cameraParams,
	cv::cuda::PtrStepSzf depthMap,
	Eigen::Matrix<float, 4, 4, Eigen::DontAlign> poseInverse,
	float minf)
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
					if (depth > 0 && depth != minf)
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
		float prevTSDF = currTSDF;
		while ((prevSign == sign) && isValid(gridSize, voxelInGridCoords))
		{
			prevTSDF = currTSDF;
			currTSDF = getFromVolume(volume, voxelInGridCoords, gridSize);

			voxelInGridCoords = currPositionInCameraWorld / VOXSIZE;
			currPositionInCameraWorld += rayStepVec;

			prevSign = sign;
			sign = currTSDF >= 0;
		}
		// printf("OUT");
		if ((sign != prevSign) && isValid(gridSize, voxelInGridCoords))
		{
			currPoint = currPositionInCameraWorld - rayStepVec * prevTSDF / (currTSDF - prevTSDF);
			;
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
		// printf("%f %f %f \n ",currPoint.x(), currPoint.y(), currPoint.z());
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

__global__ void findCorrespondencesKernel(Eigen::Matrix<float, 4, 4, Eigen::DontAlign> modelToFrame,
										  Eigen::Matrix<float, 4, 4, Eigen::DontAlign> estimatedCameraPose,
										  CameraParameters cameraParams,
										  cv::cuda::PtrStepSz<float3> surfacePoints,
										  cv::cuda::PtrStepSz<float3> surfaceNormals,
										  cv::cuda::PtrStepSz<float3> sourceVertexMap,
										  cv::cuda::PtrStepSz<float3> sourceNormalMap,
										  cv::cuda::PtrStepSz<int2> matches,
										  cv::cuda::PtrStepSzf corrDistances)
{

	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	Eigen::Matrix<float, 3, 1, Eigen::DontAlign> n, d, s;

	Eigen::Matrix<float, 4, 4, Eigen::DontAlign> estimatedFrameToFrame = modelToFrame * estimatedCameraPose;
	// printf("%f %f %f \n",estimatedCameraPose(0,0),estimatedCameraPose(1,1),estimatedCameraPose(2,2));
	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> estimatedFrametoFrameTranslation = estimatedFrameToFrame.block<3, 1>(0, 3);
	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> estimatedFrameToFrameRotation = estimatedFrameToFrame.block<3, 3>(0, 0);

	const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> estimatedFrameToModelTranslation = estimatedCameraPose.block<3, 1>(0, 3);
	const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> estimatedFrameToModelRotation = estimatedCameraPose.block<3, 3>(0, 0);

	if (0 <= x && x <= cameraParams.depthImageWidth && 0 <= y && y <= cameraParams.depthImageHeight)
	{
		Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceNormal;
		sourceNormal.x() = sourceNormalMap(y, x).x;
		sourceNormal.y() = sourceNormalMap(y, x).y;
		sourceNormal.z() = sourceNormalMap(y, x).z;

		if (!(sourceNormal.x() == 0 &&
			  sourceNormal.y() == 0 &&
			  sourceNormal.z() == 0))
		{
			Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertex;
			sourceVertex.x() = sourceVertexMap(y, x).x;
			sourceVertex.y() = sourceVertexMap(y, x).y;
			sourceVertex.z() = sourceVertexMap(y, x).z;
			//pose is passed as inverse
			Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertexPrevCamera = estimatedFrameToFrameRotation * sourceVertex + estimatedFrametoFrameTranslation;
			Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceVertexGlobal = estimatedFrameToModelRotation * sourceVertex + estimatedFrameToModelTranslation;

			//we do this part differently since here there is no iterative update step

			Eigen::Vector2i prevPixel;
			//TODO this is stolen
			prevPixel.x() = __float2int_rd(sourceVertexPrevCamera.x() * cameraParams.fovX / sourceVertexPrevCamera.z() + cameraParams.cX + 0.5f);
			prevPixel.y() = __float2int_rd(sourceVertexPrevCamera.y() * cameraParams.fovY / sourceVertexPrevCamera.z() + cameraParams.cY + 0.5f);
			if (prevPixel.x() >= 0 && prevPixel.y() >= 0 &&
				prevPixel.x() < cameraParams.depthImageWidth &&
				prevPixel.y() < cameraParams.depthImageHeight &&
				sourceVertexPrevCamera.z() >= 0)
			{
				Eigen::Matrix<float, 3, 1, Eigen::DontAlign> oldNormal;
				//TODO: Maybe apply transformation
				oldNormal.x() = surfaceNormals(prevPixel.y(), prevPixel.x()).x;
				oldNormal.y() = surfaceNormals(prevPixel.y(), prevPixel.x()).y;
				oldNormal.z() = surfaceNormals(prevPixel.y(), prevPixel.x()).z;
				if (!(oldNormal.x() == 0 &&
					  oldNormal.y() == 0 &&
					  oldNormal.z() == 0))
				{
					Eigen::Matrix<float, 3, 1, Eigen::DontAlign> oldVertex;

					oldVertex.x() = surfacePoints(prevPixel.y(), prevPixel.x()).x;
					oldVertex.y() = surfacePoints(prevPixel.y(), prevPixel.x()).y;
					oldVertex.z() = surfacePoints(prevPixel.y(), prevPixel.x()).z;
					const float distance = (oldVertex - sourceVertexGlobal).norm();
					// printf("%f \n", distance);
					if (distance <= ICP_DISTANCE_THRESHOLD)
					{

						Eigen::Matrix<float, 3, 1, Eigen::DontAlign> sourceNormalGlobal = (estimatedFrameToModelRotation * sourceNormal).normalized();
						const float sine = sourceNormalGlobal.cross(oldNormal.normalized()).norm() * 180 / M_PI;

						if (sine >= ICP_ANGLE_THRESHOLD)
						{
							// n = oldNormal;
							// d = oldVertex;
							// s = sourceVertex;
							//TODO : Make sure this is correct accessing

							// printf("%d %d  matched with %d %d     sine %f \n",y,x,prevPixel.y(),prevPixel.x(),sine);
							matches(y, x) = make_int2(prevPixel.y(), prevPixel.x());
							// corrDistances(y,x) = distance;
							corrDistances(y, x) = distance * 10.0;
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
			poseInverse,
			MINF);

		cudaDeviceSynchronize();

		cv::Mat tempResult;

		// deviceModel.download(tempResult);
		// model.setGrid(tempResult);
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

		// deviceModel.upload(model.getGrid());
		deviceSurfacePoints.upload(surfacePoints);
		deviceSurfaceNormals.upload(surfaceNormals);

		rayCastKernel<<<blocks, threads>>>(
			cameraPose,
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

		cv::imwrite("DepthImage" + std::to_string(shitCounter++) + ".png", (surfaceNormals + 1.0f) / 2.0 * 255.0f);
		model.setSurfaceNormals(surfaceNormals);
		model.setSurfacePoints(surfacePoints);
		PointCloud pcd(points, normals);
		// pcd.writeMesh("plsowrk" + std::to_string(shitCounter++) + ".off");
		model.setPointCloud(pcd);
	}

	void poseEstimation(Matrix4f &frameToModel, const CameraParameters &cameraParams, cv::cuda::GpuMat surfacePoints, cv::cuda::GpuMat surfaceNormals,
						PointCloud &inputPCD, PointCloud &initialPointCloud) // c// cv::cuda::GpuMat sourceVertexMap, cv::cuda::GpuMat sourceNormalMap)
	{

		const int threadsX = 1, threadsY = 1;
		const dim3 threads(threadsX, threadsY);
		const dim3 blocks(cameraParams.depthImageWidth /
							  threadsX,
						  cameraParams.depthImageHeight / threadsY);

		//Compute pcd vertices and normals as opencv MAT and send them to gpu

		cv::cuda::GpuMat sourceVertexMap;
		cv::cuda::GpuMat sourceNormalMap;
		cv::Mat hostSourceVertexMap(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		hostSourceVertexMap.setTo(0);

		cv::Mat hostSourceNormalMap(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		hostSourceNormalMap.setTo(0);
		cv::Mat targetPointsMat(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		targetPointsMat.setTo(0);
		cv::Mat targetNormalsMat(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC3);
		targetNormalsMat.setTo(0);
		surfacePoints.download(targetPointsMat);
		surfaceNormals.download(targetNormalsMat);
		int numPoints = inputPCD.getPoints().size();
		std::vector<Vector3f> pts = inputPCD.getPoints();
		std::vector<Vector3f> nrmls = inputPCD.getNormals();
		for (int i = 0; i < cameraParams.depthImageHeight; i++)
		{
			for (int j = 0; j < cameraParams.depthImageWidth; j++)
			{
				int index = i * cameraParams.depthImageWidth + j;
				if (pts[index].x() != MINF && nrmls[index].x() != MINF)
				{
					Vector3f pnt = pts[index];
					Vector3f normal = nrmls[index];

					hostSourceVertexMap.at<cv::Vec3f>(i, j)[0] = pnt.x();
					hostSourceVertexMap.at<cv::Vec3f>(i, j)[1] = pnt.y();
					hostSourceVertexMap.at<cv::Vec3f>(i, j)[2] = pnt.z();

					hostSourceNormalMap.at<cv::Vec3f>(i, j)[0] = normal.x();
					hostSourceNormalMap.at<cv::Vec3f>(i, j)[1] = normal.y();
					hostSourceNormalMap.at<cv::Vec3f>(i, j)[2] = normal.z();
				}
			}
		}
		

		sourceVertexMap.upload(hostSourceVertexMap);
		sourceNormalMap.upload(hostSourceNormalMap);
		cv::cuda::GpuMat matches;
		cv::Mat hostMatches(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32SC2);
		MatrixXf estimatedCameraPose = frameToModel;	//initial
		MatrixXf modelToFrame = frameToModel.inverse(); //previous frame to model
		matches.setTo(0);
		hostMatches.setTo(0);
		surfacePoints.upload(targetPointsMat);
		surfaceNormals.upload(targetNormalsMat);

		cv::Mat hostDistances(cameraParams.depthImageHeight, cameraParams.depthImageWidth, CV_32FC1);
		hostDistances.setTo(0);
		cv::cuda::GpuMat deviceDistances;
		deviceDistances.upload(hostDistances);

		static int caca = 0;

		for (int i = 0; i < ICP_ITERATIONS; i++)
		{
			hostMatches.setTo(0);
			matches.upload(hostMatches);

			hostDistances.setTo(0);
			deviceDistances.upload(hostDistances);
			cudaDeviceSynchronize();

			findCorrespondencesKernel<<<blocks, threads>>>(modelToFrame,
														   estimatedCameraPose,
														   cameraParams,
														   surfacePoints,
														   surfaceNormals,
														   sourceVertexMap,
														   sourceNormalMap,
														   matches,
														   deviceDistances);
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

			for (int i = 0; i < cameraParams.depthImageHeight; i++)
			{
				for (int j = 0; j < cameraParams.depthImageWidth; j++)
				{
					if (hostMatches.at<cv::Vec2i>(i, j)[0] != 0 && hostMatches.at<cv::Vec2i>(i, j)[1] != 0)
					{

						// cv::Vec3f color(125,255.f*i/cameraParams.depthImageHeight,255.f*j/cameraParams.depthImageWidth);

						//corr2.at<cv::Vec3f>(hostMatches.at<cv::Vec2i>(i, j)[0] ,hostMatches.at<cv::Vec2i>(i, j)[1]) = color;
						//TODO double check x-y or y-x
						int targetX = hostMatches.at<cv::Vec2i>(i, j)[0];
						int targetY = hostMatches.at<cv::Vec2i>(i, j)[1];
						Vector3f pnt;
						pnt.x() = hostSourceVertexMap.at<cv::Vec3f>(i, j)[0];
						pnt.y() = hostSourceVertexMap.at<cv::Vec3f>(i, j)[1];
						pnt.z() = hostSourceVertexMap.at<cv::Vec3f>(i, j)[2];
						Vector3f normal;
						normal.x() = hostSourceNormalMap.at<cv::Vec3f>(i, j)[0];
						normal.y() = hostSourceNormalMap.at<cv::Vec3f>(i, j)[1];
						normal.z() = hostSourceNormalMap.at<cv::Vec3f>(i, j)[2];
						sourcePts.push_back(frameToModelRotation*pnt + frameToModelTranslation);
						//sourceNormals.push_back(frameToModelRotation*normal);
						Vector3f targetPoint(targetPointsMat.at<cv::Vec3f>(targetX, targetY)[0], targetPointsMat.at<cv::Vec3f>(targetX, targetY)[1], targetPointsMat.at<cv::Vec3f>(targetX, targetY)[2]);
						Vector3f targetNormal(targetNormalsMat.at<cv::Vec3f>(targetX, targetY)[0], targetNormalsMat.at<cv::Vec3f>(targetX, targetY)[1], targetNormalsMat.at<cv::Vec3f>(targetX, targetY)[2]);
						targetNormals.push_back( targetNormal );
						targetPts.push_back(targetPoint);
						// corr1.at<cv::Vec3f>(i, j) = (rotation * srcPoint + translation - pnt).norm();
						// printf("source %d %d --> target %d %d \n",i,j,targetX,targetY);
					}
				}
			}

			if (err != cudaSuccess)
			{
				printf("CUDA Error: %s\n", cudaGetErrorString(err));
				// Possibly: exit(-1) if program cannot continue....
			}
			//estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;

			estimatedCameraPose = estimatePosePointToPlane(sourcePts, targetPts, targetNormals) * estimatedCameraPose;
		}

		deviceDistances.download(hostDistances);
		cv::imwrite("cacadistances" + std::to_string(caca++) + ".png", hostDistances * 255.0);

		std::cout << frameToModel << std::endl;
		std::cout << "***************" << std::endl;
		std::cout << estimatedCameraPose << std::endl;
		frameToModel = estimatedCameraPose;
	}
	Matrix4f estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals)
	{
		const unsigned nPoints = sourcePoints.size();

		// Build the system
		MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
		VectorXf b = VectorXf::Zero(4 * nPoints);

		for (unsigned i = 0; i < nPoints; i++)
		{
			const auto &s = sourcePoints[i];
			const auto &d = targetPoints[i];
			const auto &n = targetNormals[i];

			// TODO: [DONE] Add the point-to-plane constraints to the system
			//  1 point-to-plane row per point
			A(4 * i, 0) = n.z() * s.y() - n.y() * s.z();
			A(4 * i, 1) = n.x() * s.z() - n.z() * s.x();
			A(4 * i, 2) = n.y() * s.x() - n.x() * s.y();
			A(4 * i, 3) = n.x();
			A(4 * i, 4) = n.y();
			A(4 * i, 5) = n.z();
			b(4 * i) = n.x() * d.x() + n.y() * d.y() + n.z() * d.z() - n.x() * s.x() - n.y() * s.y() - n.z() * s.z();

			// TODO: [DONE] Add the point-to-point constraints to the system
			//  3 point-to-point rows per point (one per coordinate)
			A(4 * i + 1, 0) = 0.0f;
			A(4 * i + 1, 1) = s.z();
			A(4 * i + 1, 2) = -s.y();
			A(4 * i + 1, 3) = 1.0f;
			A(4 * i + 1, 4) = 0.0f;
			A(4 * i + 1, 5) = 0.0f;
			b(4 * i + 1) = d.x() - s.x();

			A(4 * i + 2, 0) = -s.z();
			A(4 * i + 2, 1) = 0.0f;
			A(4 * i + 2, 2) = s.x();
			A(4 * i + 2, 3) = 0.0f;
			A(4 * i + 2, 4) = 1.0f;
			A(4 * i + 2, 5) = 0.0f;
			b(4 * i + 2) = d.y() - s.y();

			A(4 * i + 3, 0) = s.y();
			A(4 * i + 3, 1) = -s.x();
			A(4 * i + 3, 2) = 0.0f;
			A(4 * i + 3, 3) = 0.0f;
			A(4 * i + 3, 4) = 0.0f;
			A(4 * i + 3, 5) = 1.0f;
			b(4 * i + 3) = d.z() - s.z();

			// TODO: [DONE] Optionally, apply a higher weight to point-to-plane correspondences
			float LAMBDA_plane = 1.0f;
			float LAMBDA_point = 0.1f;
			A(4 * i) *= LAMBDA_plane;
			b(4 * i) *= LAMBDA_plane;

			A(4 * i + 1) *= LAMBDA_point;
			b(4 * i + 1) *= LAMBDA_point;
			A(4 * i + 2) *= LAMBDA_point;
			b(4 * i + 2) *= LAMBDA_point;
			A(4 * i + 3) *= LAMBDA_point;
			b(4 * i + 3) *= LAMBDA_point;
		}

		// TODO: [DONE] Solve the system
		VectorXf x(6);

		JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
		const MatrixXf &E_i = svd.singularValues().asDiagonal().inverse();
		const MatrixXf &U_t = svd.matrixU().transpose();
		const MatrixXf &V = svd.matrixV();

		x = V * E_i * U_t * b;

		float alpha = x(0), beta = x(1), gamma = x(2);

		// Build the pose matrix
		Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
							AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
							AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

		Vector3f translation = x.tail(3);

		// TODO: [DONE] Build the pose matrix using the rotation and translation matrices
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block<3, 3>(0, 0) = rotation;
		estimatedPose.block<3, 1>(0, 3) = translation;

		return estimatedPose;
	}
} // namespace Wrapper