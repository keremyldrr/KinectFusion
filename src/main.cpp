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
#include <chrono>

#define VOXSIZE 0.01f
#define XDIM 512
#define YDIM 512
#define ZDIM 512
// #define VOXSIZE 0.005f
// #define XDIM 1024
// #define YDIM 1024
// #define ZDIM 1024

#define MIN_DEPTH 0.2f

PointCloud depthNormalMapToPcd2(const cv::Mat &vertexMap, const cv::Mat &normalMap)
{

    std::vector<Vector3f> vertices;
    std::vector<Vector3f> normals;
    std::vector<Vector4uc> colors;

    for (int i = 0; i < vertexMap.rows; i++)
    {

        for (int j = 0; j < vertexMap.cols; j++)
        {
            if (vertexMap.at<cv::Vec3f>(i, j)[0] != MINF && normalMap.at<cv::Vec3f>(i, j)[0] != MINF)
            {
                if (((!(vertexMap.at<cv::Vec3f>(i, j)[0] == 0 &&
                        vertexMap.at<cv::Vec3f>(i, j)[1] == 0 &&
                        vertexMap.at<cv::Vec3f>(i, j)[2] == 0))) &&
                    ((!(normalMap.at<cv::Vec3f>(i, j)[0] == 0 &&
                        normalMap.at<cv::Vec3f>(i, j)[1] == 0 &&
                        normalMap.at<cv::Vec3f>(i, j)[2] == 0))))
                {
                    Vector3f vert(vertexMap.at<cv::Vec3f>(i, j)[0], vertexMap.at<cv::Vec3f>(i, j)[1], vertexMap.at<cv::Vec3f>(i, j)[2]);
                    Vector3f normal(normalMap.at<cv::Vec3f>(i, j)[0], normalMap.at<cv::Vec3f>(i, j)[1], normalMap.at<cv::Vec3f>(i, j)[2]);
                    // Vector4uc color(colorMap.at<cv::Vec4b>(i, j)[0], colorMap.at<cv::Vec4b>(i, j)[1], colorMap.at<cv::Vec4b>(i, j)[2], colorMap.at<cv::Vec4b>(i, j)[3]);
                    vertices.push_back(vert);
                    normals.push_back(normal);
                    // colors.push_back(color);
                }
            }

            else
            {
                // std::cout << "MINF" << std::endl;
            }
        }
    }

    return PointCloud(vertices, normals, colors);
}
void poseEstimation(VirtualSensor &sensor, ICPOptimizer *optimizer, Matrix4f &currentCameraToWorld, Volume &model)
{

    //   PointCloud source{sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 1};

    int iters[3]{10, 5, 3};
    for (int level = 0; level >= 0; level--)
    {
        cv::Mat sf;
        cv::Mat sN;
        //model.getSurfacePoints(level).download(sf);
        //model.getSurfaceNormals(level).download(sN);
        // cv::imwrite("ANNEN.png" ,(sN + 1.0f) / 2.0 * 255.0f);
        PointCloud source = depthNormalMapToPcd2(sensor.getVertexMap(level), sensor.getNormalMap(level));
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

    const std::string filenameIn = std::string("/home/keremy/rgbd_dataset_freiburg1_xyz/");


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

    for (int i = 0; i < 1; i++)
    {

        sensor.processNextFrame();
    }   

    Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), sensor.getColorRGBX(), currentCameraToWorld.inverse());

    for (int level = 2; level >= 0; level--)
    {
        Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
    }

    int it = 0;
    Matrix4f workingPose;
    PointCloud source{sensor.getDepth(),sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 1};

    Matrix4f staticPose = Matrix4f::Identity();
    staticPose.block<3, 1>(0, 3) = Vector3f(0, 0, -1);
    while (1)
    {
        // std::vector<Vector3f> vertices;
        // std::vector<Vector4uc> colors;

        // cv::Mat volume;
        // cv::Mat colorVolume;

        // volume.setTo(0);
        // model.getGPUGrid().download(volume);
        // colorVolume.setTo(0);
        // model.getColorGPUGrid().download(colorVolume);

        // for (int i = 0; i < XDIM; i++)
        // {
        //     for (int j = 0; j < YDIM; j++)
        //     {
        //         for (int k = 0; k < ZDIM; k++)

        //         {
        //             int ind = (i * XDIM + j) * YDIM + k;
        //             assert(ind >= 0);
        //             int indFront = (i * XDIM + j) * YDIM + k + 1;
        //             int indBack = (i * XDIM + j) * YDIM + k - 1;

        //             float value = volume.at<cv::Vec2f>(ind)[0];
        //             float valueFront = volume.at<cv::Vec2f>(indFront)[0];
        //             float valueBack = volume.at<cv::Vec2f>(indBack)[0];

        //             if ( value * valueFront < 0 /*|| value * valueBack < 0*/  && value != 0)
        //             // if (abs(value) < 0.01 && value != 0)
        //             {
        //                 int vx = i - ((XDIM - 1) / 2);
        //                 int vy = j - ((YDIM - 1) / 2);
        //                 int vz = k - ((ZDIM - 1) / 2);
        //                 Vector3f voxelWorldPosition(vx + 0.5, vy + 0.5, vz + 0.5);
        //                 voxelWorldPosition *= VOXSIZE;
        //                 colors.push_back(Vector4uc(colorVolume.at<cv::Vec4b>(ind)[0],colorVolume.at<cv::Vec4b>(ind)[1],colorVolume.at<cv::Vec4b>(ind)[2],colorVolume.at<cv::Vec4b>(ind)[3]));
        //                 vertices.push_back(voxelWorldPosition);
        //             }
        //         }
        //     }
        // }

        // PointCloud pcd(vertices, vertices,colors);
        // pcd.writeMesh("tsdf_" + std::to_string(it) + ".off");
        auto sensorTime = std::chrono::high_resolution_clock::now();;
        
        sensor.processNextFrame();
        auto sensorEnd=  std::chrono::high_resolution_clock::now();
        auto t_pose_est_beg = std::chrono::high_resolution_clock::now();

        workingPose = currentCameraToWorld;
        for (int level = 0; level >= 0; level--)
        {
            bool validPose = Wrapper::poseEstimation(sensor, currentCameraToWorld, cameraParams,
                                                     model.getSurfacePoints(level), model.getSurfaceNormals(level), level);
            if (validPose)
            {
            }
            else
            {
                currentCameraToWorld = workingPose;
                // continue;
                return 0;
            }
            std::cout << "Level: " << level << std::endl;
        }
        auto t_pose_est_end = std::chrono::high_resolution_clock::now();
        auto t_pose_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_pose_est_end - t_pose_est_beg).count();

   
        auto upt_recon_beg = std::chrono::high_resolution_clock::now();
        Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), sensor.getColorRGBX(), currentCameraToWorld.inverse());
        auto upt_recon_end = std::chrono::high_resolution_clock::now();
        auto upt_recon_time = std::chrono::duration_cast<std::chrono::milliseconds>(upt_recon_end - upt_recon_beg).count();

        auto cast_beg = std::chrono::high_resolution_clock::now();
        // for (int level = 2; level >= 0; level--)
        for (int level = 0; level >= 0; level--)
        {
            Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
        }
        auto cast_end = std::chrono::high_resolution_clock::now();
        auto cast_time = std::chrono::duration_cast<std::chrono::milliseconds>(cast_end - cast_beg).count();
        auto sensor_time = std::chrono::duration_cast<std::chrono::milliseconds>(sensorEnd - sensorTime).count();

        Wrapper::rayCastStatic(model, cameraParams, staticPose, 0);

        std::cout << "POSE EST TIME: " << t_pose_time << std::endl;
        std::cout << "UPDATE TIME: " << upt_recon_time << std::endl;
        std::cout << "RAY CAST TIME: " << cast_time << std::endl;
        std::cout << "SENSOR TIME: " << sensor_time << std::endl;
        // return 0;
        auto all_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(all_end - sensorTime).count();
        std::cout << duration << "ms" << std::endl;
       
        it++;
        
    }
    return 0;
}
