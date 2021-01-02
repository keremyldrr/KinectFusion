//
// Created by kerem on 28/12/2020.
//

#ifndef KINECTFUSION_VOLUME_H
#define KINECTFUSION_VOLUME_H

#include "CameraParameters.h"
#include "PointCloud.h"
#include <vector>

struct Voxel {
    Voxel(float w = 1, float d = 0) {
        weight = w;
        distance = d;
    };
    float weight = 1;
    float distance = 0;
};


class Volume {
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


    void rayCast(const MatrixXf &cameraPose, const CameraParameters &params);

    bool pointRay(const MatrixXf &cameraPose, const CameraParameters &params, 
        int x, int y, Vector3f &surfacePoint);
};

#endif //KINECTFUSION_VOLUME_H
