#include "ICPOptimizer.h"

#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"
#include "Eigen.h"
//for cpu vision tasks like bilateral
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "kernels/include/dummy.cuh"
#include <opencv2/core/cuda.hpp>

#define VOXSIZE 0.01f
#define XDIM 512
#define YDIM 512
#define ZDIM 512

#define MIN_DEPTH 0.2f

#define MAX_WEIGHT_VALUE 128.f //inspired
void poseEstimation(VirtualSensor &sensor, ICPOptimizer *optimizer, Matrix4f &currentCameraToWorld, Volume &model)
{

    PointCloud source{sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 1};

    int iters[3]{10, 5, 3};
    for (int level = 0; level >= 0; level--)
    {
        cv::Mat sf;
        cv::Mat sN;
        // model.getSurfacePoints(level).download(sf);
        // model.getSurfaceNormals(level).download(sN);
        // cv::imwrite("ANNEN.png" ,(sN + 1.0f) / 2.0 * 255.0f);
        // PointCloud source = depthNormalMapToPcd(sensor.getVertexMap(level), sensor.getNormalMap(level));
        PointCloud target = model.getPointCloud();
        optimizer->setNbOfIterations(iters[level]);

        currentCameraToWorld = optimizer->estimatePose(source, target, currentCameraToWorld);
    }
    // Invert the transformation matrix to get the current camera pose.
    Matrix4f currentCameraPose = currentCameraToWorld.inverse();
    std::cout << "Current camera pose: " << std::endl
              << currentCameraPose << std::endl;
}
int main()
{

    // const std::string filenameIn = std::string("/home/marc/Projects/3DMotion-Scanning/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/");
    // const std::string filenameIn = std::string("/rhome/mbenedi/datasets/rgbd_dataset_freiburg1_xyz/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg3_teddy/");

    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg2_rpy/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg3_cabinet/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg2_flowerbouquet_brownbackground/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg2_coke/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg1_plant/");
    const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg1_xyz/");

    const std::string filenameBaseOut = std::string("outputMesh");

    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    Volume model(XDIM, YDIM, ZDIM, VOXSIZE, MIN_DEPTH);
    ICPOptimizer *optimizer = nullptr;
    optimizer = new LinearICPOptimizer();
    // TODO tune hyperparameters for point to plane icp
    optimizer->setMatchingMaxDistance(0.05f);
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(10);
    CameraParameters cameraParams(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
    Matrix4f currentCameraToWorld = Matrix4f::Identity();

    model.initializeSurfaceDimensions(sensor.getDepthImageHeight(), sensor.getDepthImageWidth());

    for (int i = 0; i < 74; i++)
    {
        sensor.processNextFrame();
    }

    Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());

    for (int level = 2; level >= 0; level--)
    {
        Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
    }

    int it = 0;
    Matrix4f workingPose;
    while (sensor.processNextFrame())
    {
        std::vector<Vector3f> vertices;
        cv::Mat volume;
        volume.setTo(0);
        model.getGPUGrid().download(volume);

        for (int i = 0; i < XDIM; i++)
        {
            for (int j = 0; j < XDIM; j++)
            {
                for (int k = 0; k < XDIM; k++)
                {
                    int ind = (i * XDIM + j) * XDIM + k;
                    assert(ind >= 0);

                    int indFront = (i * XDIM + j) * XDIM + k + 1;
                    int indBack = (i * XDIM + j) * XDIM + k - 1;

                    float value = volume.at<cv::Vec2f>(ind)[0];
                    float valueFront = volume.at<cv::Vec2f>(indFront)[0];
                    float valueBack = volume.at<cv::Vec2f>(indBack)[0];

                    if ((value * valueFront < 0 /*|| value * valueBack < 0*/) && value != 0)
                    // if (abs(value) < 0.01 && value != 0)
                    {
                        int vx = i - ((XDIM - 1) / 2);
                        int vy = j - ((XDIM - 1) / 2);
                        int vz = k - ((XDIM - 1) / 2);
                        Vector3f voxelWorldPosition(vx + 0.5, vy + 0.5, vz + 0.5);
                        voxelWorldPosition *= VOXSIZE;

                        vertices.push_back(voxelWorldPosition);
                    }
                }
            }
        }

        PointCloud pcd(vertices, vertices);
        pcd.writeMesh("tsdf_" + std::to_string(it++) + ".off");

        for (int level = 2; level >= 1; level--)
        {
            bool validPose = Wrapper::poseEstimation(sensor, currentCameraToWorld, cameraParams,
                                                     model.getSurfacePoints(level), model.getSurfaceNormals(level), level);
            if (validPose)
            {
                workingPose = currentCameraToWorld;
            }
            else
            {
                currentCameraToWorld = workingPose;
                // continue;
                return 0;
            }
            std::cout << "Level: " << level << std::endl;
        }

        // poseEstimation(sensor, optimizer,currentCameraToWorld,model);
        // if(it < 30){
        //    currentCameraToWorld = vladPoses[it];
        // }
        // std::cout << currentCameraToWorld << std::endl;
        Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());

        // for (int level = 2; level >= 0; level--)
        for (int level = 2; level >= 0; level--)
        {
            Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
        }
        // if (it % 1 == 0)
        // {
        //     SimpleMesh currentDepthMesh{sensor, currentCameraToWorld.inverse(), 0.1f};
        //     SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraToWorld.inverse(), 0.0015f);
        //     SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

        //     std::stringstream ss;
        //     ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
        //     std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
        //     if (!resultingMesh.writeMesh(ss.str()))
        //     {
        //         std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        //         return -1;
        //     }
        // }

        it++;
    }
    return 0;
}