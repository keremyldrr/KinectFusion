#include "ICPOptimizer.h"

#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"

//for cpu vision tasks like bilateral
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "kernels/include/dummy.cuh"
#include <opencv2/core/cuda.hpp>

#define VOXSIZE 0.005f
#define XDIM 1024
#define YDIM 1024
#define ZDIM 1024

#define MIN_DEPTH 0.2f
#define DISTANCE_THRESHOLD 2.f // inspired
#define MAX_WEIGHT_VALUE 128.f   //inspired

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
                          const MatrixXf &poseInverse)
{
    std::cout << "Updating reconstruction ..." << std::endl;

    // for (auto x = -model.gridSize.x() / 2 + 1; x < model.gridSize.x() / 2 ; x++)
    // {
    //     for (auto y = -model.gridSize.y() / 2 + 1; y < model.gridSize.y() / 2 ; y++)
    //     {
    //         for (auto z = -model.gridSize.z() / 2 + 1; z < model.gridSize.z() / 2 ; z++)
    //         {
        
    for (auto x = 0; x < model.gridSize.x()  ; x++)
    {
        for (auto y = 0; y < model.gridSize.y() ; y++)
        {
            for (auto z =0; z < model.gridSize.z() ; z++)
            {
                
                int vx = x - ((model.gridSize.x() - 1) / 2);
				int vy = y - ((model.gridSize.y() - 1) / 2);
				int vz = z - ((model.gridSize.z() - 1) / 2);

                // Our indices are between ex: [-127,128 ] ~ for 256

                // Calculate the centre of the Voxel World Pos ( Go to middle by +0.5 then multiply by voxSize)
                Vector3f voxelWorldPosition(vx + 0.5, vy + 0.5, vz + 0.5);
                voxelWorldPosition *= model.voxSize;

                const Vector3f translation = poseInverse.block<3, 1>(0, 3);
                const Matrix3f rotation = poseInverse.block<3, 3>(0, 0);

                const Vector3f voxelCamPosition = rotation * voxelWorldPosition + translation;

                if (voxelCamPosition.z() < 0)
                {
                    continue;
                }

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

                        // TODO: Consider ||t_gk-p||_2 -----> CURRENTLY ON
                        // const float value = (-1.f) * ((1.f / lambda) * (translation - voxelCamPosition).norm() - depth);
                        const float value = (-1.f) * ((1.f / lambda) * (voxelCamPosition).norm() - depth);

                        if (value >= -DISTANCE_THRESHOLD)
                        {

                            // depthMap(imagePosition.x() ,imagePosition.y()) = 255.f;
                            // TODO: Try the paper version, i.e. sign() part
                            const float sdfValue = fmin(1.f, value / DISTANCE_THRESHOLD);

                            const Voxel current = model.get(vx, vy, vz);
                            const float currValue = current.distance;
                            const float currWeight = current.weight;

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

    // std::cout << std::endl;
    // cv::Mat eben = model.getGrid();
	// 	std::vector<Vector3f> surfacePoints;
	// 		std::vector<Vector3f> surfaceNormals;
	// 		for(int i=0;i<512;i++){
	// 			for(int j=0;j<512;j++){
	// 				for(int k=0;k<512;k++){
	// 					if(abs(eben.at<cv::Vec2f>(i,j,k)[0]) < 0.1 && abs(eben.at<cv::Vec2f>(i,j,k)[0] !=0))
	// 					{
	// 						surfacePoints.push_back(Vector3f(i,j,k));
	// 					}
	
	// 				}
	// 			}
	// 		}
	// 		std::cout << surfacePoints.size()<<" valid voxels" << std::endl;
	// 		PointCloud pcd(surfacePoints,surfaceNormals);
	// 		pcd.writeMesh("YARAK3s.off");

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

    // const std::string filenameIn = std::string("/home/marc/Projects/3DMotion-Scanning/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/");
    const std::string filenameIn = std::string("/rhome/mbenedi/datasets/rgbd_dataset_freiburg1_xyz/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg1_xyz/");

    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg1_xyz/");
    const std::string filenameBaseOut = std::string("outputMesh");

    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    ICPOptimizer *optimizer = nullptr;
    optimizer = new LinearICPOptimizer();
    // TODO tune hyperparameters for point to plane icp
    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(10);

    Volume model(XDIM, YDIM, ZDIM, VOXSIZE, MIN_DEPTH);
    std::vector<Matrix4f> estimatedPoses;
    CameraParameters cameraParams(sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

    // Processing the first frame as a reference (to initialize structures)
    sensor.processNextFrame();
    PointCloud initialPointCloud(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(),
                                 sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    estimatedPoses.push_back(currentCameraToWorld.inverse());

    // updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());
    Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());
    Wrapper::rayCast(model, cameraParams, currentCameraToWorld);
    // model.rayCast(currentCameraToWorld, cameraParams);

    int i = 1;
    while (sensor.processNextFrame() && i < 20)
    {
        poseEstimation(sensor, optimizer, currentCameraToWorld, initialPointCloud, estimatedPoses);
        Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());

        // updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());
        Wrapper::rayCast(model, cameraParams, currentCameraToWorld);
        // model.rayCast(currentCameraToWorld, cameraParams);
        estimatedPoses.push_back(currentCameraToWorld.inverse());

        if (i % 1 == 0)
        {
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

    // VISUALIZATION OF TSDF

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
