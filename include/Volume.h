//
// Created by kerem on 28/12/2020.
//

#ifndef KINECTFUSION_VOLUME_H
#define KINECTFUSION_VOLUME_H

#include "CameraParameters.h"
#include "PointCloud.h"
#include <vector>
#include <opencv2/opencv.hpp>
struct Voxel
{
    Voxel(float w = 0, float d = 1)
    {
        weight = w;
        distance = d;
    };
    float weight = 0;
    float distance = 1;
};

class Volume
{
private:
    Voxel *grid;
    PointCloud pcd;
    const float minimumDepth;

public:
    const Vector3i gridSize;
    const float voxSize;

    Volume(int xdim, int ydim, int zdim, float voxelSize, float minDepth);

    ~Volume();

    PointCloud getPointCloud();
    void setPointCloud(PointCloud &pointCloud);

    const Voxel *get(int x, int y, int z);
    void set(int x, int y, int z, const Voxel &value);

    void rayCast(const MatrixXf &cameraPose, const CameraParameters &params,std::vector<cv::Point3d> &rays);

    bool pointRay(const MatrixXf &cameraPose, const CameraParameters &params,
                  int x, int y, Vector3f &surfacePoint,Vector3f &surfaceNormal,std::vector<cv::Point3d> &rays);

    bool isValid(const Vector3f & point) {
        return point.x() < gridSize.x() / 2 &&
           point.y() < gridSize.y() / 2 && 
           point.z() < gridSize.z() / 2 && 
           point.x() > -gridSize.x() / 2 &&
           point.y() > -gridSize.y() / 2 && 
           point.z() > -gridSize.z() / 2;
    }
};

#endif //KINECTFUSION_VOLUME_H
