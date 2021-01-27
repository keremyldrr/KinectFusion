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
    Voxel(float w = 0, float d = 0)
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
    //TODO remove this get
    const float get(int i){

        return grid[i].distance;

    }
    void set(int x, int y, int z, const Voxel &value);

    void rayCast(const MatrixXf &cameraPose, const CameraParameters &params,std::vector<cv::Point3d> &rays);

    bool pointRay(const MatrixXf &cameraPose, const CameraParameters &params,
                  int x, int y, Vector3f &surfacePoint,Vector3f &surfaceNormal,std::vector<cv::Point3d> &rays);

    bool isValid(const Vector3f & point);

    float interpolation(const Vector3f &position);
};

#endif //KINECTFUSION_VOLUME_H
