#pragma once
#include "Eigen.h"

struct CameraParameters
{
    CameraParameters(const Matrix3f &depthIntrinsics, int imageWidth, int imageHeight,float scaleFactor)
    {
       	float fovX = depthIntrinsics(0, 0)*scaleFactor;
		float fovY = depthIntrinsics(1, 1)*scaleFactor;
		float cX = (depthIntrinsics(0, 2) + 0.5f)*scaleFactor - 0.5;
		float cY = (depthIntrinsics(1, 2) + 0.5f)*scaleFactor - 0.5f;
        depthImageWidth = imageWidth;
        depthImageHeight = imageHeight;
    };
    float fovX;
    float fovY;
    float cX;
    float cY;
    int depthImageWidth;
    int depthImageHeight;
};