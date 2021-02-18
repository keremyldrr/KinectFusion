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
    std::vector<cv::cuda::GpuMat> surfacePoints;
    std::vector<cv::cuda::GpuMat> surfaceNormals;
    cv::cuda::GpuMat gpuGrid;

public:
    const Vector3i gridSize;
    const float voxSize;

    Volume(int xdim, int ydim, int zdim, float voxelSize, float minDepth);

    ~Volume();

    PointCloud getPointCloud();
    void setPointCloud(PointCloud &pointCloud);
    cv::cuda::GpuMat & getSurfacePoints(int i)
    {
        return surfacePoints[i];
    }

    void initializeSurfaceDimensions(int h,int w)
    {
        for (int level = 0; level < 3; level++)
        {
            float scale = pow(0.5,level);
            cv::Mat temp(h*scale,w*scale,CV_32FC3);
            temp.setTo(0);
            cv::Mat tempNormal(h*scale,w*scale,CV_32FC3);
            tempNormal.setTo(0);
            cv::cuda::GpuMat tempGpu;
            cv::cuda::GpuMat tempGpuNormal;
            tempGpu.upload(temp);
            tempGpuNormal.upload(tempNormal);
            surfacePoints.push_back(tempGpu);
            surfaceNormals.push_back(tempGpuNormal);                
        }
    }
    void setSurfacePoints(cv::Mat &sP, int i)
    {
        surfacePoints[i].upload(sP);
    }
    cv::cuda::GpuMat & getSurfaceNormals(int i)
    {
        return surfaceNormals[i];
    }
    void setSurfaceNormals(cv::Mat &sN, int i)
    {
        surfaceNormals[i].upload(sN);
    }
    const Voxel get(int x, int y, int z);
    //TODO remove this get
    cv::cuda::GpuMat getGPUGrid()
    {
        return gpuGrid;
    }

    void set(int x, int y, int z, const Voxel &value);

    bool isValid(const Vector3f &point);
};

#endif //KINECTFUSION_VOLUME_H
