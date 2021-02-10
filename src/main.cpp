#include "ICPOptimizer.h"

#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"

//for cpu vision tasks like bilateral
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "kernels/include/dummy.cuh"
#include <opencv2/core/cuda.hpp>

#define VOXSIZE 0.005
#define XDIM 1024
#define YDIM 1024
#define ZDIM 1024

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


    sensor.processNextFrame();
    Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());
    
    for (int level = 2 ; level >= 0; level--)
        Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);

    while (sensor.processNextFrame())
    {
        // ***************************************************
        // *********************GPU***************************
        // ***************************************************
        for (int level = 2; level >= 0; level--)
            Wrapper::poseEstimation(sensor, currentCameraToWorld, cameraParams, model.getSurfacePoints(level), model.getSurfaceNormals(level),level);
        Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());
        
        for (int level = 2; level >= 0; level--)
            Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);

    }
    return 0;
}
