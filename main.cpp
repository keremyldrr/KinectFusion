#include <iostream>
#include "Volume.h"
#include "VirtualSensor.h"
#include "ICPOptimizer.h"


//for cpu vision tasks like bilateral

#define VOXSIZE 2

#define XDIM 256

#define YDIM 256

#define ZDIM 256

#define MIN_DEPTH 0

#define DISTANCE_THRESHOLD 25.f //stolen
#define MAX_WEIGHT_VALUE 128 //stolen


void surfacePrediction(Volume &model) {
    // model.rayCast();

}

void updateReconstruction(Volume &model,const CameraParameters &cameraParams,const float * const depthMap,const MatrixXf poseInverse) {
#pragma omp parallel for
    for (auto x = 0; x < model.size.x(); x++) {
        for (auto y = 0; y < model.size.y(); y++) {
            for (auto z = 0; z < model.size.z(); z++) {
                const Vector3f voxelWorldPosition((x + 0.5f) * model.voxSize,
                                                  (y + 0.5f) * model.voxSize,
                                                  (z + 0.5f) * model.voxSize);

                const Vector3f voxelCamPosition = poseInverse * voxelWorldPosition; //voxel in camera coordinate frame
                const Vector2i imagePosition(
                        voxelCamPosition.x() / voxelCamPosition.z() * cameraParams.fovX + cameraParams.cX,
                        voxelCamPosition.y() / voxelCamPosition.z() * cameraParams.fovY + cameraParams.cY);
                const float depth = depthMap[imagePosition.x() * cameraParams.depthImageWidth + imagePosition.y()];
                if (depth <= 0 || depth == MINF)
                    continue;
                const Vector3f homogenImagePosition((imagePosition.x() - cameraParams.cX) / cameraParams.fovX,
                                                    (imagePosition.y() - cameraParams.cY) / cameraParams.fovY,
                                                    1.0f);
                const float lambda = homogenImagePosition.norm();

                const float value = (-1.f) * ((1.f / lambda) * voxelCamPosition.norm() - depth);
                if (value >= -DISTANCE_THRESHOLD) {

                    const float sdfValue = fmin(1.f, value / DISTANCE_THRESHOLD);
                    const Voxel *current = model.get(x, y, z);
                    const float currValue = current->distance;
                    const float currWeight = current->weight;

                    const float addWeight = 1;

                    const float nextTSDF = (currWeight * currValue + addWeight * sdfValue) /
                                           (currWeight + addWeight);
                    Voxel newVox(nextTSDF, fmin(currWeight + addWeight, MAX_WEIGHT_VALUE));
                    model.set(x, y, z, newVox);
                }

            }
        }
    }
}



void poseEstimation(VirtualSensor &sensor, ICPOptimizer *optimizer,Matrix4f &currentCameraToWorld,const PointCloud &target,std::vector<Matrix4f> &estimatedPoses){

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
//    sensor.processNextFrame();

    //PointCloud model(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(),
     //                sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
    std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();

    estimatedPoses.push_back(currentCameraToWorld.inverse());
    //surface measurement
    //
    CameraParameters cameraParams(sensor.getDepthIntrinsics(),sensor.getDepthImageWidth(),sensor.getDepthImageHeight());

    Volume model(XDIM, YDIM, ZDIM, VOXSIZE, MIN_DEPTH);
    while( sensor.processNextFrame()) {
        //surface measurement

        poseEstimation(sensor, optimizer, currentCameraToWorld, model.getPointCloud(), estimatedPoses);

        updateReconstruction(model,cameraParams,sensor.getDepth(),currentCameraToWorld.inverse());
        surfacePrediction(model);
    }
    delete optimizer;
	return 0;
}

