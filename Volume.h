//
// Created by kerem on 28/12/2020.
//

#ifndef KINECTFUSION_VOLUME_H
#define KINECTFUSION_VOLUME_H

#include "CameraParameters.h"
#include "PointCloud.h"
#include <vector>

struct Voxel {
    Voxel(float w = 1, float d = 0) {
        weight = w;
        distance = d;
    };
    float weight = 1;
    float distance = 0;
};


class Volume {
public:
    Volume(int xdim, int ydim, int zdim, int vSize, float minimumDepth) {

        grid = new Voxel[xdim * ydim * zdim];
        size = Vector3i(xdim, ydim, zdim);
        voxSize = vSize;
        this->minimumDepth = minimumDepth;
    }


    Vector3i size;
    int voxSize;

    ~Volume() {
        delete grid;
    }

    PointCloud getPointCloud() {

    }

    const Voxel *get(int x, int y, int z) {

        return &grid[x * size.x() + y * size.y() + z];
        //return grid[z *(x*size.x() + y)];

    }

    void set(int x, int y, int z, const Voxel &value) {
        grid[x * size.x() + y * size.y() + z].distance = value.distance;
        grid[x * size.x() + y * size.y() + z].weight = value.weight;

    }


    void rayCast(const MatrixXf &cameraPose, const CameraParameters &params) {
        // TODO: Search for possible optimizations here...
        // minimumDepth = 0 ==> for now
        std::vector<Vector3f> surface_points;

        // loop over all the pixels
        for (int y = 0; y < params.depthImageHeight; y++) {
            for (int x = 0; x < params.depthImageWidth; x++) {
                Vector3f curr_P;
                bool exists = pointRay(cameraPose, params, x, y, minimumDepth, curr_P);
                if (exists) surface_points.push_back(curr_P);

            }
        }

    }

    bool pointRay(const MatrixXf &cameraPose, const CameraParameters &params, int x, int y, float minimumDepth,
                  Vector3f &surf_point) {
        Vector3f pix_camera_coord;
        // Point in camera pose
        pix_camera_coord << (x - params.cX) / params.fovX,
                (y - params.cY) / params.fovY,
                1.0;
        Vector3f curr_position = pix_camera_coord.normalized() * minimumDepth;
        // starting location
        curr_position -= cameraPose.block<3, 1>(0, 3);
        // direction vec / step vector
        Vector3f step_vec = pix_camera_coord.normalized() * this->voxSize;

        step_vec = cameraPose.block<3, 3>(0, 0) * step_vec;

        // TODO: Whatever the fuck is this, needs to be calculated
        float min_step = this->voxSize / 2.0f;
        Vector3f voxel_loc = curr_position + Vector3f(min_step, min_step, min_step);
        // TODO: ***************************************************

        float curr_TSDF = this->get((int) voxel_loc.x(),
                                    (int) voxel_loc.y(),
                                    (int) voxel_loc.z())->distance;

        // TODO: Interpolation for points...

        // TODO: Is it necessary to check voxel_loc.x.y.z < 0 ????
        while (curr_TSDF > 0 && voxel_loc.x() < this->size.x() &&
               voxel_loc.y() < this->size.y() && voxel_loc.z() < this->size.z()) {

            // TODO: Whatever the fuck is this, needs to be calculated
            voxel_loc = curr_position + Vector3f(min_step, min_step, min_step);
            // TODO: ***************************************************

            curr_position += step_vec;

            curr_TSDF = this->get((int) voxel_loc.x(),
                                  (int) voxel_loc.y(),
                                  (int) voxel_loc.z())->distance;

        }

        // TODO: check if voxel_loc.x.y.z inside grid
        if (curr_TSDF > -1 && voxel_loc.x() < this->size.x() &&
            voxel_loc.y() < this->size.y() && voxel_loc.z() < this->size.z()) {
            surf_point = curr_position;
            return true;
        }
        return false;
    }


private:
    Voxel *grid;
    PointCloud pcd;
    float minimumDepth;

};


#endif //KINECTFUSION_VOLUME_H
