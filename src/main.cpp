#include "ICPOptimizer.h"

#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"

//for cpu vision tasks like bilateral
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#define VOXSIZE 0.005f

#define XDIM 1024

#define YDIM 1024

#define ZDIM 1024

#define MIN_DEPTH 0.2f

#define DISTANCE_THRESHOLD 2.f // inspired
#define MAX_WEIGHT_VALUE 128   //inspired

// Class Debug Sphere for easier testing 
class DebugSphere
{
private:
    Vector3f center;
    float radius;

public:
    DebugSphere(Vector3f c, float r) : center(c), radius(r){};

    bool isInside(const Vector3f &point)
    {
        float distance = pow(center.x() - point.x(), 2) + pow(center.y() - point.y(), 2) + pow(center.z() - point.z(), 2) - radius;
        return distance < 0;
    }
};



void updateReconstruction(Volume &model,
                          const CameraParameters &cameraParams,
                          const float *const depthMap,
                          const MatrixXf &poseInverse, std::vector<cv::Point3d> &negPts, std::vector<cv::Point3d> &posPts)
{
    std::cout << "Updating reconstruction ..." << std::endl;
#pragma omp parallel for
    for (auto x = 0; x < model.gridSize.x(); x++)
    {
        std::cout << x << "/" << model.gridSize.x() << "\r";
        for (auto y = 0; y < model.gridSize.y(); y++)
        {
            for (auto z = 0; z < model.gridSize.z(); z++)
            {

                // Our indices are between ex: [-127,128 ] ~ for 256
                int vx = x - (model.gridSize.x() - 1) / 2;
                int vy = y - (model.gridSize.y() - 1) / 2;
                int vz = z - (model.gridSize.z() - 1) / 2;

                // Calculate the centre of the Voxel World Pos ( Go to middle by +0.5 then multiply by voxSize)
                Vector3f voxelWorldPosition(vx + 0.5, vy + 0.5, vz + 0.5);
                voxelWorldPosition *= model.voxSize;

                const Vector3f translation = poseInverse.block<3, 1>(0, 3);
                const Matrix3f rotation = poseInverse.block<3, 3>(0, 0);


                const Vector3f voxelCamPosition = rotation * voxelWorldPosition + translation;

                if (voxelCamPosition.z() < 0)
                    continue;

                /* Code for debugging done with the Debugsphere */

                // DebugSphere sph1(Vector3f(0, 10, 30), 50);
                // DebugSphere sph2(Vector3f(0, 0, 30), 50);

                // Voxel newVox;
                // if (sph1.isInside(Vector3f(vx, vy, vz)) || sph2.isInside(Vector3f(vx, vy, vz))  )
                //     newVox.distance = -1;
                // else
                //     newVox.distance = 1;
                // model.set(vx, vy, vz, newVox);
                // continue;



                const Vector2i imagePosition(
                    (voxelCamPosition.y() / voxelCamPosition.z()) * cameraParams.fovY + cameraParams.cY,
                    (voxelCamPosition.x() / voxelCamPosition.z()) * cameraParams.fovX + cameraParams.cX);

                if (!(imagePosition.x() < 0 || imagePosition.x() >= cameraParams.depthImageHeight ||
                      imagePosition.y() < 0 || imagePosition.y() >= cameraParams.depthImageWidth))
                {

                    const float depth = depthMap[imagePosition.x() * cameraParams.depthImageWidth + imagePosition.y()];

                    if (depth > 0 && depth != MINF)
                    {
                        const Vector3f homogenImagePosition(
                            (imagePosition.x() - cameraParams.cX) / cameraParams.fovX,
                            (imagePosition.y() - cameraParams.cY) / cameraParams.fovY,
                            1.0f);
                        const float lambda = homogenImagePosition.norm();
                        
                        // TODO: Consider ||T_gk-p||_2
                        const float value = (-1.f) * ((1.f / lambda) * voxelCamPosition.norm() - depth);

                        if (value >= -DISTANCE_THRESHOLD)
                        {
                            // TODO: Try the paper version, i.e. sign() part
                            const float sdfValue = fmin(1.f, value / DISTANCE_THRESHOLD);
                            const Voxel *current = model.get(vx, vy, vz);
                            const float currValue = current->distance;
                            const float currWeight = current->weight;
 
                            const float addWeight = 1;
                            const float nextTSDF = 
                                (currWeight * currValue + addWeight * sdfValue) /
                                 (currWeight + addWeight);
                            Voxel newVox;
                            newVox.distance = nextTSDF;
                            // TODO: Check the MAX_WEIGHT_VALUE and how it would work after max iterations
                            newVox.weight = fmin(currWeight + addWeight, MAX_WEIGHT_VALUE);

                            
                            model.set(vx, vy, vz, newVox);
                        }
                    }
                }
            }
        }
    }
    std::cout << std::endl;
}

void poseEstimation(VirtualSensor &sensor, ICPOptimizer *optimizer, Matrix4f &currentCameraToWorld, const PointCloud &target, std::vector<Matrix4f> &estimatedPoses)
{

    float *depthMap = sensor.getDepth();
    Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
    Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

    // Estimate the current camera pose from source to target mesh with ICP optimization.
    // We downsample the source image to speed up the correspondence matching.
    PointCloud source{sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 8};
    currentCameraToWorld = optimizer->estimatePose(source, target, currentCameraToWorld);

    // Invert the transformation matrix to get the current camera pose.
    Matrix4f currentCameraPose = currentCameraToWorld.inverse();
    std::cout << "Current camera pose: " << std::endl
              << currentCameraPose << std::endl;
    estimatedPoses.push_back(currentCameraPose);
}


int main()
{
    //initialize sensor
    // Marc's Linux settings
    // const std::string filenameIn = std::string("/home/marc/Projects/3DMotion-Scanning/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/");
    std::string filenameIn = std::string("/rhome/mbenedi/rgbd_dataset_freiburg1_xyz/");
    std::string filenameBaseOut = std::string("halfcaca");

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }


    ICPOptimizer *optimizer = nullptr;
    optimizer = new LinearICPOptimizer();

    //TODO tune hyperparameters for point to plane icp
    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(10);

    // We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
    sensor.processNextFrame();

    PointCloud initialPointCloud(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(),
                                 sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

    cv::Mat depthImageGT(sensor.getColorImageHeight(), sensor.getColorImageWidth(), CV_32FC3);
    std::vector<Vector3f> gtNormals = initialPointCloud.getNormals();

    for (auto x = 0; x < sensor.getDepthImageHeight(); x++)
    {
        for (auto y = 0; y < sensor.getDepthImageWidth(); y++)
        {
            // Sad normals visualization
            depthImageGT.at<cv::Vec3b>(x, y)[0] = abs(gtNormals[x * sensor.getDepthImageWidth() + y].x()) * 255;
            depthImageGT.at<cv::Vec3b>(x, y)[1] = abs(gtNormals[x * sensor.getDepthImageWidth() + y].y()) * 255;
            depthImageGT.at<cv::Vec3b>(x, y)[2] = abs(gtNormals[x * sensor.getDepthImageWidth() + y].z()) * 255;
        }
    }
    // cv::imshow("DEPTHNORMALS",depthImageGT);
    // cv::waitKey(0);
    std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();

    estimatedPoses.push_back(currentCameraToWorld.inverse());

    CameraParameters cameraParams(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

    Volume model(XDIM, YDIM, ZDIM, VOXSIZE, MIN_DEPTH);
    // model.setPointCloud(initialPointCloud);
    // saveToMesh(sensor, currentCameraToWorld, "caca");
    std::vector<cv::Point3d> posPts;
    std::vector<cv::Point3d> negPts;
    std::vector<cv::Point3d> rays;
    std::vector<cv::Point3d> verts;
    std::vector<cv::Point3d> verts2;

    // cv::viz::Viz3d window; //creating a Viz window
                           //Displaying the Coordinate Origin (0,0,0)
    // window.showWidget("coordinate", cv::viz::WCoordinateSystem(12));

    updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse(), negPts, posPts);

    model.rayCast(currentCameraToWorld, cameraParams, rays);

    int i = 1;
    while (sensor.processNextFrame() && i <20)
    {
        //surface measurement
        poseEstimation(sensor, optimizer, currentCameraToWorld,initialPointCloud, estimatedPoses);
        updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse(), negPts, posPts);
        model.rayCast(currentCameraToWorld, cameraParams, rays);

        estimatedPoses.push_back(currentCameraToWorld.inverse());
        // model.rayCast(currentCameraToWorld,cameraParams);

        if (i % 1 == 0)
        {
            // // For Checking ICP correction
            // // saveToMesh(sensor, currentCameraToWorld, "caca");
            // SimpleMesh currentDepthMesh{sensor, currentCameraToWorld.inverse(), 0.1f};
            // SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraToWorld.inverse(), 0.0015f);
            // SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

            // std::stringstream ss;
            // ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
            // std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
            // if (!resultingMesh.writeMesh(ss.str()))
            // {
            //     std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
            //     return -1;
            // }
        }
        i += 1;
    }
    auto elems = model.getPointCloud();

    std::cout << "Finished" << std::endl;
    std::vector<Vector3f> points = elems.getPoints();
    for (int i = 0; i < elems.getPoints().size(); i++)
    {
        //change of basis
        Vector3f voxelInGridCoords = (points[i]) / model.voxSize;
        verts.push_back(cv::Point3d(voxelInGridCoords.x(), voxelInGridCoords.y(), voxelInGridCoords.z()));
    }

    //TODO VISUALIZATION OF TSDF
    // std::vector<unsigned char> distances;
    // std::vector<cv::Point3d> gridPoints;
    // for(int x= 0; x<XDIM;x++){
    //     for(int y= 0; y<YDIM;y++){
    //         for(int z= 0; z<ZDIM;z++){
    //             int vx = x - (model.gridSize.x() - 1) / 2;
    //             int vy = y - (model.gridSize.y() - 1) / 2;
    //             int vz = z - (model.gridSize.z() - 1) / 2;
    //             // distances[x + model.gridSize.x() * (y + model.gridSize.z() * z)] = model.get(vx,vy,vz)->distance;
    //             float value = model.get(vx,vy,vz)->distance;
    //             if(abs(value) < 0.1){
    //             gridPoints.push_back(cv::Point3d(vx,vy,vz));
    //             distances.push_back((unsigned char ) (value * 255));
    //             }
    //         }
    //     }
    // }

    // cv::Mat dists(1,distances.size(),CV_8UC1,&distances[0]);
    // cv::Mat im_color;
    // cv::applyColorMap(dists, im_color, cv::COLORMAP_HOT	); 
    // // delete optimizer;
    //  window.showWidget("bluwsad", cv::viz::WCloud(gridPoints, im_color));
    // window.showWidget("bluwsad", cv::viz::WCloud(verts,cv::viz::Color::blue() ));
    // window.showWidget("points", cv::viz::WCloud(posPts, c));
    // window.showWidget("points2", cv::viz::WCloud(negPts, cv::viz::Color::red()));

    // window.spin();

    return 0;
}
