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
														const MatrixXf &poseInverse);
} // namespace Wrapper