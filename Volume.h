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
    float minimumDepth;

public:
    Vector3i gridSize;
    float voxSize;

    Volume(int xdim, int ydim, int zdim, float voxelSize, float minDepth) {
        grid = new Voxel[xdim * ydim * zdim];
        gridSize = Vector3i(xdim, ydim, zdim);
        voxSize = voxelSize;
        minimumDepth = minDepth;
    }

    ~Volume() {
        delete grid;
    }

    PointCloud getPointCloud() {

    }

    const Voxel *get(int x, int y, int z) {
        return &grid[x * gridSize.x() + y * gridSize.y() + z];
    }

    void set(int x, int y, int z, const Voxel &value) {
        grid[x * gridSize.x() + y * gridSize.y() + z].distance = value.distance;
        grid[x * gridSize.x() + y * gridSize.y() + z].weight = value.weight;
    }


    void rayCast(const MatrixXf &cameraPose, const CameraParameters &params) {
        // TODO: Search for possible optimizations here...
        // minimumDepth = 0 ==> for now
        std::vector<Vector3f> surfacePoints;

        for (int y = 0; y < params.depthImageHeight; y++) {
            for (int x = 0; x < params.depthImageWidth; x++) {
                Vector3f currPoint;
                bool exists = pointRay(cameraPose, params, x, y, currPoint);
                if (exists) {
                    surfacePoints.push_back(currPoint);
                }
            }
        }
    }

    bool pointRay(const MatrixXf &cameraPose, const CameraParameters &params, 
        int x, int y, Vector3f &surfacePoint) {
        Vector3f pixelInCameraCoords;
        pixelInCameraCoords << (x - params.cX) / params.fovX,
                (y - params.cY) / params.fovY,
                1.0;
        Vector3f currPosition = pixelInCameraCoords.normalized() * minimumDepth;
        // starting location
        currPosition -= cameraPose.block<3, 1>(0, 3);
        // direction vec / step vector
        Vector3f stepVec = pixelInCameraCoords.normalized() * this->voxSize;

        stepVec = cameraPose.block<3, 3>(0, 0) * stepVec;

        // TODO: Whatever the fuck is this, needs to be calculated
        float minStep = this->voxSize / 2.0f;
        Vector3f voxelInCameraCoords = currPosition + Vector3f(minStep, minStep, minStep);
        // TODO: ***************************************************

        float currTSDF = this->get((int) voxelInCameraCoords.x(),
                                    (int) voxelInCameraCoords.y(),
                                    (int) voxelInCameraCoords.z())->distance;

        // TODO: Interpolation for points...

        // TODO: Is it necessary to check voxelInCameraCoords.x.y.z < 0 ????
        while (currTSDF > 0 && voxelInCameraCoords.x() < gridSize.x() &&
               voxelInCameraCoords.y() < gridSize.y() && voxelInCameraCoords.z() < gridSize.z()) {

            // TODO: Whatever the fuck is this, needs to be calculated
            voxelInCameraCoords = currPosition + Vector3f(minStep, minStep, minStep);
            // TODO: ***************************************************

            currPosition += stepVec;

            currTSDF = this->get((int) voxelInCameraCoords.x(),
                                  (int) voxelInCameraCoords.y(),
                                  (int) voxelInCameraCoords.z())->distance;

        }

        // TODO: check if voxelInCameraCoords.x.y.z inside grid
        if (currTSDF > -1 && voxelInCameraCoords.x() < gridSize.x() &&
            voxelInCameraCoords.y() < gridSize.y() && voxelInCameraCoords.z() < gridSize.z()) {
            surfacePoint = currPosition;
            return true;
        }
        return false;
    }

};


#endif //KINECTFUSION_VOLUME_H
