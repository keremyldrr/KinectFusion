#pragma once
#include "Eigen.h"

struct CameraParameters
{
    CameraParameters(const Matrix3f &depthIntrinsics, int imageWidth, int imageHeight)
    {
        fovX = depthIntrinsics(0, 0);
        fovY = depthIntrinsics(1, 1);
        cX = depthIntrinsics(0, 2);
        cY = depthIntrinsics(1, 2);
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