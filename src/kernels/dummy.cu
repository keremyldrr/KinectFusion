#include <cuda_runtime.h>

__device__ __forceinline__ 
float dummyFunc(
  // const float3 &volume_max, const Vec3fda &origin, const Vec3fda &direction
  )
{
  // float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
  // float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
  // float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

  // return fmin(fmin(txmax, tymax), tzmax);
}

__global__ 
void dummyKernel(
  // const PtrStepSz<short2> tsdf_volume, const PtrStepSz<uchar3> color_volume,
  //                                   PtrStepSz<float3> model_vertex, PtrStepSz<float3> model_normal,
  //                                   PtrStepSz<uchar3> model_color,
  //                                   const int3 volume_size, const float voxel_scale,
  //                                   const CameraParameters cam_parameters,
  //                                   const float truncation_distance,
  //                                   const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
  //                                   const Vec3fda translation
                                    )
{
  // const int x = blockIdx.x * blockDim.x + threadIdx.x;
  // const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // if (x >= model_vertex.cols || y >= model_vertex.rows)
  //   return;
  dummyFunc();
  
}

void dummyEntrypoint(
  // const VolumeData &volume,
  //                       GpuMat &model_vertex, GpuMat &model_normal, GpuMat &model_color,
  //                       const CameraParameters &cam_parameters,
  //                       const float truncation_distance,
  //                       const Eigen::Matrix4f &pose
                        )
{
  // model_vertex.setTo(0);
  // model_normal.setTo(0);
  // model_color.setTo(0);

  // dim3 threads(32, 32);
  // dim3 blocks((model_vertex.cols + threads.x - 1) / threads.x,
  //             (model_vertex.rows + threads.y - 1) / threads.y);

  dummyKernel<<<1, 1>>>();
  // raycast_tsdf_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.color_volume,
  //                                          model_vertex, model_normal, model_color,
  //                                          volume.volume_size, volume.voxel_scale,
  //                                          cam_parameters,
  //                                          truncation_distance,
  //                                          pose.block(0, 0, 3, 3), pose.block(0, 3, 3, 1));

  // cudaThreadSynchronize();
}
