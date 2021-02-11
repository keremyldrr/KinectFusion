#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <opencv2/core/cuda.hpp>

#include "Volume.h"
namespace Wrapper
{
	void wrapper(cv::cuda::GpuMat &img, Volume &model);
	void updateReconstruction(Volume &model,
														const CameraParameters &cameraParams,
														const float *const depthMap,
														const MatrixXf &modelToFrame);
	void rayCast(Volume &model,
							 CameraParameters cameraParams,
							 const MatrixXf &frameToModel,int level);

	bool poseEstimation(VirtualSensor &sensor,Matrix4f &modelToFramePose, CameraParameters cameraParams, cv::cuda::GpuMat surfacePoints, cv::cuda::GpuMat surfaceNormals,int level);
	Matrix4f estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals);

} // namespace Wrapper