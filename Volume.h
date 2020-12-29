//
// Created by kerem on 28/12/2020.
//

#ifndef KINECTFUSION_VOLUME_H
#define KINECTFUSION_VOLUME_H
#include "Eigen.h"
#include "PointCloud.h"
#include <vector>
struct Voxel{
    short weight = 1;
    short distance = 0;
};
class Volume {
public:
    Volume(int xdim,int ydim,int zdim, int voxSize){

        grid = new Voxel[xdim * ydim * zdim];
    }

    Voxel *grid;

    ~Volume(){
        delete grid;
    }
    PointCloud getPointCloud(){

    }
    void rayCast()
    {
        pcd;

    }
private:
    PointCloud pcd;
};


#endif //KINECTFUSION_VOLUME_H
