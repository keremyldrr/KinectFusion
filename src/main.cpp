#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"
//#include "ICPOptimizer.h"
//for cpu vision tasks like bilateral
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#define VOXSIZE 0.01f

#define XDIM 256

#define YDIM 256

#define ZDIM 256

#define MIN_DEPTH 0.2f

#define DISTANCE_THRESHOLD 25.f //stolen
#define MAX_WEIGHT_VALUE 128    //stolen

void surfacePrediction(Volume &model)
{
}

void updateReconstruction(Volume &model,
                          const CameraParameters &cameraParams,
                          const float *const depthMap,
                          const MatrixXf &poseInverse,std::vector<cv::Point3d> &negPts,std::vector<cv::Point3d> &posPts)
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
                
                /*
                * The origin of our 3D world (0,0,0) (Camera position in the reference frame) is at the center of our grid
                */
                // TODO: Extract this into Volume.h maybe
                int vx=x - (model.gridSize.x() - 1)/2;
                int vy=y - (model.gridSize.y()-1)/2;
                int vz=z - (model.gridSize.z()-1)/2;
                // const Vector3f shiftWorldCenterToVoxelCoords(
                //     model.gridSize.x() / 2,
                //     model.gridSize.y() / 2,
                //     model.gridSize.z() / 2);
                // // TODO: Consider change of basis
                // Vector3f voxelWorldPosition(
                //     (x + 0.5f) ,
                //     (y + 0.5f) ,
                //     (z + 0.5f) );
                // voxelWorldPosition -= shiftWorldCenterToVoxelCoords;
                Vector3f voxelWorldPosition(vx,vy,vz);
                voxelWorldPosition*= model.voxSize;
                // TODO: Rename translation and rotation
                // TODO: Check names poseInverse, voxelWorldPosition, voxelCamPosition
                const Vector3f translation = poseInverse.block<3, 1>(0, 3);
                const Matrix3f rotation = poseInverse.block<3, 3>(0, 0);
                // std::cout << poseInverse << std::endl;
                // const Vector3f voxelCamPosition = poseInverse * voxelWorldPosition;
                const Vector3f voxelCamPosition =rotation * voxelWorldPosition + translation;
                // std::cout << voxelCamPosition << std::endl;
                if(voxelCamPosition.z() < 0)
                    continue;
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
                        const float value = (-1.f) * ((1.f / lambda) * voxelCamPosition.norm() - depth);

                        if (value >= -DISTANCE_THRESHOLD)
                        {
                            const float sdfValue = fmin(1.f, value / DISTANCE_THRESHOLD);
                            const Voxel *current = model.get(vx, vy, vz);
                            const float currValue = current->distance;
                            const float currWeight = current->weight;
                            const float addWeight = 1;
                            const float nextTSDF =
                                (currWeight * currValue + addWeight * sdfValue) / (currWeight + addWeight);
                            Voxel newVox;
                            newVox.distance = nextTSDF;
                            newVox.weight = fmin(currWeight + addWeight, MAX_WEIGHT_VALUE);
                            model.set(vx, vy, vz, newVox);
                            // if(newVox.distance != 0)
                           //std::cout << nextTSDF << std::endl;
                            posPts.push_back(cv::Point3d(vx,vy,vz));
                        }
                       

                    }
                     else{
                            negPts.push_back(cv::Point3d(vx,vy,vz));
                        }
                }
            }
        }
    }
    std::cout << std::endl;
}

// void poseEstimation(VirtualSensor &sensor, ICPOptimizer *optimizer, Matrix4f &currentCameraToWorld, const PointCloud &target, std::vector<Matrix4f> &estimatedPoses)
// {

//     float *depthMap = sensor.getDepth();
//     Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
//     Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

//     // Estimate the current camera pose from source to target mesh with ICP optimization.
//     // We downsample the source image to speed up the correspondence matching.
//     PointCloud source{sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 8};
//     currentCameraToWorld = optimizer->estimatePose(source, target, currentCameraToWorld);

//     // Invert the transformation matrix to get the current camera pose.
//     Matrix4f currentCameraPose = currentCameraToWorld.inverse();
//     std::cout << "Current camera pose: " << std::endl
//               << currentCameraPose << std::endl;
//     estimatedPoses.push_back(currentCameraPose);
// }

// TODO: Create a mesh from the TSDF
int saveToMesh(VirtualSensor &sensor, const Matrix4f &currentCameraPose, const std::string &filenameBaseOut)
{
    // We write out the mesh to file for debugging.
    SimpleMesh currentDepthMesh{sensor, currentCameraPose, 0.1f};
    SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
    SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

    std::stringstream ss;
    ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
    std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
    if (!resultingMesh.writeMesh(ss.str()))
    {
        std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        return -1;
    }
    return 0;
}

// ICPOptimizer *initializeICP()
// {
//     ICPOptimizer *optimizer = nullptr;
//     optimizer = new LinearICPOptimizer();

//     //TODO tune hyperparameters for point to plane icp
//     optimizer->setMatchingMaxDistance(0.1f);
//     optimizer->usePointToPlaneConstraints(true);
//     optimizer->setNbOfIterations(10);

//     return optimizer;
// }

int main()
{
    //initialize sensor

    //const std::string filenameIn = std::string("/home/marc/Projects/3DMotion-Scanning/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/");
     std::string filenameIn = std::string("../../rgbd_dataset_freiburg1_xyz/");
    std::string filenameBaseOut = std::string("halfcaca");

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }
    //ICPOptimizer *optimizer = initializeICP();
    cv::Mat im(XDIM,YDIM,CV_64F);
    // ICPOptimizer *optimizer = nullptr;
    // optimizer = new LinearICPOptimizer();

    // //TODO tune hyperparameters for point to plane icp
    // optimizer->setMatchingMaxDistance(0.1f);
    // optimizer->usePointToPlaneConstraints(true);
    // optimizer->setNbOfIterations(10);

    // We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
    sensor.processNextFrame();

    PointCloud initialPointCloud(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(),
                                 sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
    std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();

    estimatedPoses.push_back(currentCameraToWorld.inverse());
    //surface measurement

    CameraParameters cameraParams(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

    //TODO: Delete this counter
    int i = 0;

    Volume model(XDIM, YDIM, ZDIM, VOXSIZE, MIN_DEPTH);
    //model.setPointCloud(initialPointCloud);
   // saveToMesh(sensor, currentCameraToWorld, "caca");
    std::vector<cv::Point3d> posPts;
    std::vector<cv::Point3d> negPts;
    std::vector<cv::Point3d> rays;
    std::vector<cv::Point3d> verts;
    std::vector<cv::Point3d> verts2;



    posPts.push_back(cv::Point3d(128,128,128));

    posPts.push_back(cv::Point3d(0,0,0));

    posPts.push_back(cv::Point3d(256,256,256));

    posPts.push_back(cv::Point3d(256,0,0));

    posPts.push_back(cv::Point3d(0,256,0));
    posPts.push_back(cv::Point3d(256,256,0));
    posPts.push_back(cv::Point3d(0,0,256));
    posPts.push_back(cv::Point3d(0,256,256));
    posPts.push_back(cv::Point3d(256,0,256));


    cv::viz::Viz3d window; //creating a Viz window
//Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", cv::viz::WCoordinateSystem(128));
//Displaying the 3D points in green
    updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld,negPts,posPts);
    model.rayCast(currentCameraToWorld,cameraParams,rays);

    // window.showWidget("points", cv::viz::WCloud(posPts, cv::viz::Color::green()));
    // window.showWidget("points2", cv::viz::WCloud(negPts, cv::viz::Color::red()));
    // auto elems =  initialPointCloud;//model.getPointCloud();;
    auto elems = model.getPointCloud();;

    std::vector<Vector3f> points = elems.getPoints();
    for(int i = 0;i< elems.getPoints().size();i++)
    {
    const Vector3f shiftWorldCenterToVoxelCoords(
        model.gridSize.x() / 2,
        model.gridSize.y() / 2,
        model.gridSize.z() / 2);

    //change of basis
    Vector3f voxelInGridCoords = (points[i])/model.voxSize;
        verts.push_back(cv::Point3d(voxelInGridCoords.x(),voxelInGridCoords.y(),voxelInGridCoords.z()));
    }
    // auto elems =  
    auto initElems = initialPointCloud;//model.getPointCloud();;

    std::vector<Vector3f> points2 = initElems.getPoints();
    for(int i = 0;i< initElems.getPoints().size();i++)
    {
    const Vector3f shiftWorldCenterToVoxelCoords(
        model.gridSize.x() / 2,
        model.gridSize.y() / 2,
        model.gridSize.z() / 2);

    //change of basis
    Vector3f voxelInGridCoords = (points2[i])/model.voxSize;
        verts2.push_back(cv::Point3d(voxelInGridCoords.x(),voxelInGridCoords.y(),voxelInGridCoords.z()));
    }


    // window.showWidget("bluw", cv::viz::WCloud(rays, cv::viz::Color::yellow()));
    window.showWidget("asdsad", cv::viz::WCloud(verts, cv::viz::Color::blue()));
    window.showWidget("gg", cv::viz::WCloud(verts2, cv::viz::Color::green()));

    window.spin();

    while (sensor.processNextFrame() && i < 50)
    {
        //surface measurement
            // poseEstimation(sensor, optimizer, currentCameraToWorld, initialPointCloud, estimatedPoses);
            break;
        
            estimatedPoses.push_back(currentCameraToWorld.inverse());
     
           // model.rayCast(currentCameraToWorld,cameraParams);

           
                if(i  % 5 == 0){
                    //SAVE TO MESH IS BROKEN
                // saveToMesh(sensor, currentCameraToWorld, "caca");
                    SimpleMesh currentDepthMesh{sensor, currentCameraToWorld.inverse(), 0.1f};
                    SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraToWorld.inverse(), 0.0015f);
                        SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

                        std::stringstream ss;
                        ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
                        std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
                        if (!resultingMesh.writeMesh(ss.str()))
                        {
                            std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
                            return -1;
                        }
                                    
                }
                i += 1;
    }
    // delete optimizer;
  
    return 0;
}