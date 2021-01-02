//
// Created by kerem on 28/12/2020.
//
#include "Volume.h"
#include "CameraParameters.h"
#include "PointCloud.h"
#include <vector>

Volume::Volume(int xdim, int ydim, int zdim, float voxelSize, float minDepth) : voxSize(voxelSize), gridSize(Vector3i(xdim, ydim, zdim)), minimumDepth(minDepth)
{
    grid = new Voxel[xdim * ydim * zdim];
}

Volume::~Volume()
{
    delete grid;
}

// TODO: Return reference to pointcloud not a copy
PointCloud Volume::getPointCloud()
{
    return pcd;
}

void Volume::setPointCloud(PointCloud &pointCloud)
{
    pcd = pointCloud;
}

const Voxel *Volume::get(int x, int y, int z)
{
    return &grid[x * gridSize.x() + y * gridSize.y() + z];
}

void Volume::set(int x, int y, int z, const Voxel &value)
{
    grid[x * gridSize.x() + y * gridSize.y() + z].distance = value.distance;
    grid[x * gridSize.x() + y * gridSize.y() + z].weight = value.weight;
}

void Volume::rayCast(const MatrixXf &cameraPose, const CameraParameters &params)
{
    // TODO: Search for possible optimizations here...
    std::vector<Vector3f> surfacePoints;

    for (int y = 0; y < params.depthImageHeight; y++)
    {
        for (int x = 0; x < params.depthImageWidth; x++)
        {
            Vector3f currPoint;
            bool exists = pointRay(cameraPose, params, x, y, currPoint);
            if (exists)
            {
                surfacePoints.push_back(currPoint);
            }
        }
    }
}
bool Volume::pointRay(const MatrixXf &cameraPose, const CameraParameters &params,
                      int x, int y, Vector3f &surfacePoint)
{
    const Vector3f pixelInCameraCoords(
        (x - params.cX) / params.fovX,
        (y - params.cY) / params.fovY,
        1.0);

    Vector3f currPositionInCameraWorld = pixelInCameraCoords.normalized() * minimumDepth;
    // TODO: Check if - or +
    // Translate point to 3D world
    currPositionInCameraWorld -= cameraPose.block<3, 1>(0, 3);

    Vector3f rayStepVec = pixelInCameraCoords.normalized() * voxSize;
    // Rotate rayStepVec to 3D world
    rayStepVec = cameraPose.block<3, 3>(0, 0) * rayStepVec;

    const Vector3f shiftWorldCenterToVoxelCoords(
        gridSize.x() / 2,
        gridSize.y() / 2,
        gridSize.z() / 2);
    Vector3f voxelInGridCoords = (currPositionInCameraWorld + shiftWorldCenterToVoxelCoords) / voxSize;

    // TODO: Interpolation for points
    float currTSDF = get((int)voxelInGridCoords.x(),
                         (int)voxelInGridCoords.y(),
                         (int)voxelInGridCoords.z())
                         ->distance;

    // TODO: Is it necessary to check voxelInGridCoords.x.y.z < 0 ?
    // TODO: Change of sign in both direction - -> + and + -> -
    while (currTSDF > 0 && voxelInGridCoords.x() < gridSize.x() &&
           voxelInGridCoords.y() < gridSize.y() && voxelInGridCoords.z() < gridSize.z())
    {

        voxelInGridCoords = (currPositionInCameraWorld + shiftWorldCenterToVoxelCoords) / voxSize;
        currPositionInCameraWorld += rayStepVec;

        // TODO: Interpolation for points...
        currTSDF = get((int)voxelInGridCoords.x(),
                       (int)voxelInGridCoords.y(),
                       (int)voxelInGridCoords.z())
                       ->distance;
    }

    // TODO: Is it necessary to check voxelInGridCoords.x.y.z < 0 ?
    if (currTSDF > -1 && voxelInGridCoords.x() < gridSize.x() &&
        voxelInGridCoords.y() < gridSize.y() && voxelInGridCoords.z() < gridSize.z())
    {
        surfacePoint = currPositionInCameraWorld;
        return true;
    }
    return false;
}
