//
// Created by kerem on 28/12/2020.
//

#ifndef KINECTFUSION_VOLUME_H
#define KINECTFUSION_VOLUME_H
#include "Eigen.h"
#include "PointCloud.h"
#include <vector>
struct Voxel {
    Voxel(float w, float d) {
        weight = w;
        distance = d;
    };
    float weight = 1;
    float distance = 0;
};


class Volume {
public:
    Volume(int xdim, int ydim, int zdim, int vSize) {

        grid = new Voxel[xdim * ydim * zdim];
        size = Vector3i(xdim, ydim, zdim);
        voxSize = vSize;
    }


    Vector3i size;
    int voxSize;

    ~Volume() {
        delete grid;
    }

    PointCloud getPointCloud() {

    }

    const Voxel *get(int x, int y, int z) {

        return &grid[x * size.x() + y * size.y() + z];
        //return grid[z *(x*size.x() + y)];

    }

    void set(int x, int y, int z, const Voxel &value) {
        grid[x * size.x() + y * size.y() + z].distance = value.distance;
        grid[x * size.x() + y * size.y() + z].weight = value.weight;

    }

    void rayCast() {
        pcd;

    }

private:
    Voxel *grid;
    PointCloud pcd;

};


#endif //KINECTFUSION_VOLUME_H
