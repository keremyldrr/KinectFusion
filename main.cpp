#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
//for cpu vision tasks like bilateral
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#define USE_POINT_TO_PLANE	1
#define USE_LINEAR_ICP		1

#define RUN_SEQUENCE_ICP	1

void preprocess(PointCloud &p){


    //bilateral filter  here

}
void estimatePoseWithICP(VirtualSensor &sensor, ICPOptimizer *optimizer,Matrix4f &currentCameraToWorld,PointCloud &target,std::vector<Matrix4f> &estimatedPoses){

    float* depthMap = sensor.getDepth();
    Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
    Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

    // Estimate the current camera pose from source to target mesh with ICP optimization.
    // We downsample the source image to speed up the correspondence matching.
    PointCloud source{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 8 };
    currentCameraToWorld = optimizer->estimatePose(source, target, currentCameraToWorld);

    // Invert the transformation matrix to get the current camera pose.
    Matrix4f currentCameraPose = currentCameraToWorld.inverse();
    std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
    estimatedPoses.push_back(currentCameraPose);

}


//int reconstructRoom() {
//
//	// Setup the optimizer.
//
//	int i = 0;
//	const int iMax = 50;
//	while (sensor.processNextFrame() && i <= iMax) {
//
//
//		if (i % 5 == 0) {
//			// We write out the mesh to file for debugging.
//			SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
//			SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
//			SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());
//
//			std::stringstream ss;
//			ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
//			std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
//			if (!resultingMesh.writeMesh(ss.str())) {
//				std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
//				return -1;
//			}
//		}
//
//		i++;
//	}
//
//	delete optimizer;
//
//	return 0;
//}
ICPOptimizer * initializeICP(){
    ICPOptimizer* optimizer = nullptr;
    optimizer = new LinearICPOptimizer();

    //TODO tune hyperparameters for point to plane icp
    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(10);

    return optimizer;

}
int main() {
    //initialize sensor

    std::string filenameIn = std::string("../../rgbd_dataset_freiburg1_xyz/");
    std::string filenameBaseOut = std::string("mesh_");

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }
    ICPOptimizer *optimizer = initializeICP();
    // We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
    sensor.processNextFrame();
    std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    estimatedPoses.push_back(currentCameraToWorld.inverse());
    //surface measurement
    PointCloud target{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight() };


    estimatePoseWithICP(sensor,optimizer,currentCameraToWorld,target,estimatedPoses);
    // initialize TSDF voxel grid
    //frame by frame

    // We store the estimated camera poses.


    //update reconstruction
    //predict surface using raycasting

    //filter the point cloud subsample and construct pyramid
    // estimate pose

    delete optimizer;
	return 0;
}
