#include "ICPOptimizer.h"

#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"

//for cpu vision tasks like bilateral
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "kernels/include/dummy.cuh"
#include <opencv2/core/cuda.hpp>

#define VOXSIZE 0.01
#define XDIM 512
#define YDIM 512
#define ZDIM 512

#define MIN_DEPTH 0.2f
#define DISTANCE_THRESHOLD 2.f // inspired
#define MAX_WEIGHT_VALUE 128.f //inspired

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

    CameraParameters cameraParams(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
    Matrix4f currentCameraToWorld = Matrix4f::Identity();

    model.initializeSurfaceDimensions(sensor.getDepthImageHeight(), sensor.getDepthImageWidth());

    for (int i = 0; i < 1; i++)
    {
        sensor.processNextFrame();
    }
    
    Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());

    for (int level = 2; level >= 0; level--)
    {
        Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
    }

    // while (true)
    int it = 0;
    Matrix4f workingPose;
    while (sensor.processNextFrame())
    {
        // std::vector<Vector3f> vertices;
        // cv::Mat volume;
        // volume.setTo(0);
        // model.getGPUGrid().download(volume);

        // for (int i = 0; i < 1024; i++)
        // {
        //     for (int j = 0; j < 1024; j++)
        //     {
        //         for (int k = 0; k < 1024; k++)
        //         {
        //             int ind = (i * 1024 + j) * 1024 + k;
        //             assert(ind >= 0);

        //             int indFront = (i * 1024 + j) * 1024 + k + 1;
        //             int indBack = (i * 1024 + j) * 1024 + k - 1;

        //             float value = volume.at<cv::Vec2f>(ind)[0];
        //             float valueFront = volume.at<cv::Vec2f>(indFront)[0];
        //             float valueBack = volume.at<cv::Vec2f>(indBack)[0];

        //             if ((value * valueFront < 0 /*|| value * valueBack < 0*/) && value != 0)
        //             // if (abs(value) < 0.01 && value != 0)
        //             {
        //                 int vx = i - ((1024 - 1) / 2);
        //                 int vy = j - ((1024 - 1) / 2);
        //                 int vz = k - ((1024 - 1) / 2);
        //                 Vector3f voxelWorldPosition(vx + 0.5, vy + 0.5, vz + 0.5);
        //                 voxelWorldPosition *= VOXSIZE;

        //                 vertices.push_back(voxelWorldPosition);
        //             }
        //         }
        //     }
        // }

        // PointCloud pcd(vertices, vertices);
        // pcd.writeMesh("tsdf_" + std::to_string(it++) + ".off");

        for (int level = 2; level >= 0; level--)
        {
            bool validPose = Wrapper::poseEstimation(sensor, currentCameraToWorld, cameraParams,
                                    model.getSurfacePoints(level), model.getSurfaceNormals(level), level);
            if(validPose) {
                workingPose = currentCameraToWorld;
            } else {
                // currentCameraToWorld = workingPose;
                // continue;
                return 0;
            }
            std::cout << "Level: " << level << std::endl;
        }

        Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());

        for (int level = 2; level >= 0; level--)
        {
            Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
        }
    }
    return 0;
}
