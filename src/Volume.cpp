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
  //grid = new Voxel[size];


  std::vector< int> sizes{(int)size,(unsigned int) 1};
  std::cout << sizes[0];
  // ? TODO: Now weights may be initialized at 0 also

  //TODO MAKE THIS PROPER CONSTRUCTOR
  grid = cv::Mat(sizes, CV_32FC2); //.reshape(2, sizes);
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
  // pointCloud.writeMesh("ANAN.off");

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

  // grid.at<cv::Vec2f>(x, y, z)[0] = value.distance;
  // grid.at<cv::Vec2f>(x, y, z)[1] = value.weight;
  grid.at<cv::Vec2f>(ind)[0] = value.distance;
  grid.at<cv::Vec2f>(ind)[1] = value.weight;
}

bool Volume::isValid(const Vector3f &point)
{
  return point.x() < gridSize.x() / 2 && point.y() < gridSize.y() / 2 &&
         point.z() < gridSize.z() / 2 && point.x() > -gridSize.x() / 2 &&
         point.y() > -gridSize.y() / 2 && point.z() > -gridSize.z() / 2;
}

float Volume::interpolation(const Vector3f &position)
{

  Vector3f pointInGrid((int)position.x(), (int)position.y(), (int)position.z());

  // Toggle to disable interpolation
  // return get((int)position.x(), (int)position.y(), (int)position.z()).distance;

  Vector3f voxelCenter(pointInGrid.x() + 0.5f, pointInGrid.y() + 0.5f,
                       pointInGrid.z() + 0.5f);

  pointInGrid.x() = (position.x() < voxelCenter.x()) ? (pointInGrid.x() - 1)
                                                     : pointInGrid.x();
  pointInGrid.y() = (position.y() < voxelCenter.y()) ? (pointInGrid.y() - 1)
                                                     : pointInGrid.y();
  pointInGrid.z() = (position.z() < voxelCenter.z()) ? (pointInGrid.z() - 1)
                                                     : pointInGrid.z();

  // pointInGrid = Vector3f(pointInGrid.x() - 1, pointInGrid.y() - 1,
  // pointInGrid.z() - 1);

  // Check Distance correctness
  const float distX = abs((abs(position.x()) - abs((pointInGrid.x()) + 0.5f)));
  const float distY = abs((abs(position.y()) - abs((pointInGrid.y()) + 0.5f)));
  const float distZ = abs((abs(position.z()) - abs((pointInGrid.z()) + 0.5f)));

  // TODO: Check the correctness of below, just a sanity check
  return (isValid(Vector3f(pointInGrid.x(), pointInGrid.y(), pointInGrid.z()))
              ? get(pointInGrid.x(), pointInGrid.y(), pointInGrid.z()).distance
              : 0.0f) *
             (1 - distX) * (1 - distY) * (1 - distZ) +
         (isValid(
              Vector3f(pointInGrid.x(), pointInGrid.y(), pointInGrid.z() + 1))
              ? get(pointInGrid.x(), pointInGrid.y(), pointInGrid.z() + 1)
                    .distance
              : 0.0f) *
             (1 - distX) * (1 - distY) * (distZ) +
         (isValid(
              Vector3f(pointInGrid.x(), pointInGrid.y() + 1, pointInGrid.z()))
              ? get(pointInGrid.x(), pointInGrid.y() + 1, pointInGrid.z())
                    .distance
              : 0.0f) *
             (1 - distX) * distY * (1 - distZ) +
         (isValid(Vector3f(pointInGrid.x(), pointInGrid.y() + 1,
                           pointInGrid.z() + 1))
              ? get(pointInGrid.x(), pointInGrid.y() + 1, pointInGrid.z() + 1)
                    .distance
              : 0.0f) *
             (1 - distX) * distY * distZ +
         (isValid(
              Vector3f(pointInGrid.x() + 1, pointInGrid.y(), pointInGrid.z()))
              ? get(pointInGrid.x() + 1, pointInGrid.y(), pointInGrid.z())
                    .distance
              : 0.0f) *
             distX * (1 - distY) * (1 - distZ) +
         (isValid(Vector3f(pointInGrid.x() + 1, pointInGrid.y(),
                           pointInGrid.z() + 1))
              ? get(pointInGrid.x() + 1, pointInGrid.y(), pointInGrid.z() + 1)
                    .distance
              : 0.0f) *
             distX * (1 - distY) * distZ +
         (isValid(Vector3f(pointInGrid.x() + 1, pointInGrid.y() + 1,
                           pointInGrid.z()))
              ? get(pointInGrid.x() + 1, pointInGrid.y() + 1, pointInGrid.z())
                    .distance
              : 0.0f) *
             distX * distY * (1 - distZ) +
         (isValid(Vector3f(pointInGrid.x() + 1, pointInGrid.y() + 1,
                           pointInGrid.z() + 1))
              ? get(pointInGrid.x() + 1, pointInGrid.y() + 1,
                    pointInGrid.z() + 1)
                    .distance
              : 0.0f) *
             distX * distY * distZ;
}

void Volume::rayCast(const MatrixXf &cameraPose, const CameraParameters &params)
{

  // TODO: Search for possible optimizations here...
  std::vector<Vector3f> surfacePoints;
  std::vector<Vector3f> surfaceNormals;
  float fovX = params.fovX;
  float fovY = params.fovY;
  float cX = params.cX;
  float cY = params.cY;
  static int num = 0;
  cv::Mat depthImage((int)params.depthImageHeight, (int)params.depthImageWidth,
                     CV_32FC3); // CV_8UC3
  depthImage = 0;
  for (int x = 0; x < params.depthImageHeight; x++)
  {
    for (int y = 0; y < params.depthImageWidth; y++)
    {
      Vector3f currPoint, currNormal;
      bool exists =
          pointRay(cameraPose, params, y, x, currPoint, currNormal);
      if (exists)
      {
        surfacePoints.push_back(currPoint);
        surfaceNormals.push_back(currNormal);
        currNormal.normalize();

        depthImage.at<cv::Vec3f>(x, y)[0] = currNormal.x();
        depthImage.at<cv::Vec3f>(x, y)[1] = currNormal.y();
        depthImage.at<cv::Vec3f>(x, y)[2] = currNormal.z();
      }
      else
      {
        // surfacePoints.push_back(Vector3f(MINF, MINF, MINF));
      }
    }
  }
  std::cout << "Surface predicted with " << surfacePoints.size()
            << " vertices \n";
  cv::imwrite("DepthImage" + std::to_string(num) + ".png", (depthImage + 1.0) / 2.0 * 255.0);
  // cv::imshow("s", depthImage);
  //cv::waitKey(0);
  auto PP = PointCloud(surfacePoints, surfaceNormals);
  setPointCloud(PP);
  pcd.writeMesh("pcd" + std::to_string(num) + ".off");
  num++;
}

bool Volume::pointRay(const MatrixXf &cameraPose,
                      const CameraParameters &params, int x, int y,
                      Vector3f &surfacePoint, Vector3f &currNormal)
{

  const Vector3f pixelInCameraCoords((x - params.cX) / params.fovX,
                                     (y - params.cY) / params.fovY, 1.0);

  Vector3f currPositionInCameraWorld = pixelInCameraCoords.normalized() * minimumDepth;

  currPositionInCameraWorld += cameraPose.block<3, 1>(0, 3);
  Vector3f rayStepVec = pixelInCameraCoords.normalized() * voxSize;
  // Rotate rayStepVec to 3D world
  rayStepVec = (cameraPose.block<3, 3>(0, 0) * rayStepVec);

  Vector3f voxelInGridCoords = currPositionInCameraWorld / voxSize;

  float currTSDF = 1.0;
  float prevTSDF = currTSDF;
  bool sign = true;
  bool prevSign = sign;

  int maxRayDist = 1000;

  while ((prevSign == sign) && isValid(voxelInGridCoords))
  {
    currTSDF = get((int)voxelInGridCoords.x(),
                   (int)voxelInGridCoords.y(),
                   (int)voxelInGridCoords.z())
                   .distance;

    voxelInGridCoords = currPositionInCameraWorld / voxSize;
    currPositionInCameraWorld += rayStepVec;

    prevTSDF = currTSDF;
    prevSign = sign;
    sign = currTSDF >= 0;
  }

  if ((sign != prevSign) && isValid(voxelInGridCoords))
  {
    surfacePoint = currPositionInCameraWorld;
  }
  else
  {
    return false;
  }

  Vector3f neighbor = voxelInGridCoords;
  neighbor.x() += 1;
  if (!isValid(neighbor))
    return false;
  const float Fx1 = interpolation(neighbor);

  neighbor = voxelInGridCoords;

  neighbor.x() -= 1;
  if (!isValid(neighbor))
    return false;
  const float Fx2 = interpolation(neighbor);

  currNormal.x() = (Fx1 - Fx2);

  neighbor = voxelInGridCoords;
  neighbor.y() += 1;
  if (!isValid(neighbor))
    return false;
  const float Fy1 = interpolation(neighbor);

  neighbor = voxelInGridCoords;
  neighbor.y() -= 1;
  if (!isValid(neighbor))
    return false;
  const float Fy2 = interpolation(neighbor);

  currNormal.y() = (Fy1 - Fy2);

  neighbor = voxelInGridCoords;
  neighbor.z() += 1;
  if (!isValid(neighbor))
    return false;
  const float Fz1 = interpolation(neighbor);

  neighbor = voxelInGridCoords;
  neighbor.z() -= 1;
  if (!isValid(neighbor))
    return false;
  const float Fz2 = interpolation(neighbor);

  currNormal.z() = (Fz1 - Fz2);

  if (currNormal.norm() == 0)
    return false;

  currNormal.normalize();

  return true;
}
