#include "Volume.h"
#include "CameraParameters.h"
#include "PointCloud.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

Volume::Volume(int xdim, int ydim, int zdim, float voxelSize, float minDepth)
    : voxSize(voxelSize), gridSize(Vector3i(xdim, ydim, zdim)),
      minimumDepth(minDepth)
{
  unsigned long long size = (unsigned long long)xdim * (unsigned long long)ydim * (unsigned long long)zdim;

  std::vector< int> sizes{(int)size,(unsigned int) 1};
  // ? TODO: Now weights may be initialized at 0 also

  //TODO MAKE THIS PROPER CONSTRUCTOR
  grid = cv::Mat(sizes, CV_32FC2);
  gpuGrid.upload(grid);
  
}

Volume::~Volume()
{
  // delete grid;
}

// TODO: Return reference to pointcloud not a copy
PointCloud Volume::getPointCloud() { return pcd; }

void Volume::setPointCloud(PointCloud &pointCloud)
{
  pcd = pointCloud;
}

const Voxel Volume::get(int x, int y, int z)
{
  // -127 128
  x += (gridSize.x() - 1) / 2;
  y += (gridSize.y() - 1) / 2;
  z += (gridSize.z() - 1) / 2;

  int ind = (x * gridSize.y() + y) * gridSize.z() + z;

  Voxel value(
      grid.at<cv::Vec2f>(ind)[0],
      grid.at<cv::Vec2f>(ind)[1]);
  return value;
}

void Volume::set(int x, int y, int z, const Voxel &value)
{
  x += (gridSize.x() - 1) / 2;
  y += (gridSize.y() - 1) / 2;
  z += (gridSize.z() - 1) / 2;

  int ind = (x * gridSize.y() + y) * gridSize.z() + z;

  grid.at<cv::Vec2f>(ind)[0] = value.distance;
  grid.at<cv::Vec2f>(ind)[1] = value.weight;
}

bool Volume::isValid(const Vector3f &point)
{
  return point.x() < gridSize.x() / 2 && point.y() < gridSize.y() / 2 &&
         point.z() < gridSize.z() / 2 && point.x() > -gridSize.x() / 2 &&
         point.y() > -gridSize.y() / 2 && point.z() > -gridSize.z() / 2;
}
