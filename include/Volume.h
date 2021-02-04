#ifndef KINECTFUSION_VOLUME_H
#define KINECTFUSION_VOLUME_H

#include "CameraParameters.h"
#include "PointCloud.h"
#include <vector>
#include <opencv2/opencv.hpp>

struct Voxel
{
    Voxel(float d = 0, float w = 0)
    {
        weight = w;
        distance = d;
    };
    float weight = 0;
    float distance = 0;
};

class Volume
{
private:
    cv::Mat grid;
    PointCloud pcd;
    const float minimumDepth;
    cv::cuda::GpuMat surfacePoints;
    cv::cuda::GpuMat surfaceNormals;
    cv::cuda::GpuMat gpuGrid;
public:
    const Vector3i gridSize;
    const float voxSize;

    Volume(int xdim, int ydim, int zdim, float voxelSize, float minDepth);

    ~Volume();

    PointCloud getPointCloud();
    void setPointCloud(PointCloud &pointCloud);
    cv::cuda::GpuMat getSurfacePoints()
    {
        return surfacePoints;
    }
    void setSurfacePoints(cv::Mat sP)
    {
        surfacePoints.upload(sP);
    }
    cv::cuda::GpuMat getSurfaceNormals()
    {

        return surfaceNormals;
    }
    void setSurfaceNormals(cv::Mat sN)
    {
        surfaceNormals.upload(sN);
    }
    const Voxel get(int x, int y, int z);
    //TODO remove this get
    cv::cuda::GpuMat getGPUGrid()
    {
        return gpuGrid;
    }
  
    void set(int x, int y, int z, const Voxel &value);

    void rayCast(const MatrixXf &cameraPose, const CameraParameters &params);

    bool pointRay(
        const MatrixXf &cameraPose, const CameraParameters &params,
        int x, int y, Vector3f &surfacePoint, Vector3f &surfaceNormal);

    bool isValid(const Vector3f &point);

    float interpolation(const Vector3f &position);
};

#endif //KINECTFUSION_VOLUME_H
