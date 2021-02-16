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

#define VOXSIZE 0.005f
#define XDIM 1024
#define YDIM 1024
#define ZDIM 1024

#define MIN_DEPTH 0.2f

#define MAX_WEIGHT_VALUE 128.f //inspired

int main()
{
    Matrix4f temp;
    std::vector<Matrix4f> vladPoses;
    temp << 0.999987, 0.00237673, -0.00461295, -0.00389506,
        -0.00236889, 0.999996, 0.00170378, 0.00209172,
        0.00461698, -0.00169283, 0.999988, 0.0123907,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999929, 0.00608992, -0.0102131, -0.00669156,
        -0.00603122, 0.999965, 0.00576886, 0.0083944,
        0.0102478, -0.00570685, 0.999931, 0.0252492,
        0, 0, 0, 1;
    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999855, 0.00657683, -0.0157467, -0.00850759,
        -0.00628311, 0.999807, 0.0186305, 0.00891627,
        0.0158661, -0.0185288, 0.999703, 0.0378223,
        0, 0, 0, 1;
    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999765, 0.00584117, -0.0208951, -0.00937592,
        -0.00524783, 0.999585, 0.0283393, 0.0104637,
        0.0210519, -0.028223, 0.99938, 0.0502662,
        0, 0, 0, 1;
    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999697, 0.00432939, -0.0242479, -0.010813,
        -0.00339693, 0.999259, 0.0383654, 0.00971457,
        0.024396, -0.0382715, 0.99897, 0.0620759,
        0, 0, 0, 1;
    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999629, 0.000684856, -0.0272309, -0.0122543,
        0.000464783, 0.99911, 0.0421894, 0.0113955,
        0.0272356, -0.0421865, 0.998739, 0.0737708,
        0, 0, 0, 1;
    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999563, -0.000361093, -0.0295855, -0.0143967,
        0.00170832, 0.998962, 0.0455241, 0.0136204,
        0.0295384, -0.0455547, 0.998525, 0.0853921,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999444, 0.000196888, -0.0333441, -0.0171011,
        0.00141497, 0.998832, 0.0483097, 0.0169713,
        0.0333147, -0.0483301, 0.998276, 0.0976272,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999186, 0.000995969, -0.0403306, -0.0178394,
        0.00127052, 0.998423, 0.0561331, 0.0223296,
        0.0403229, -0.0561387, 0.997608, 0.110357,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999039, 0.00271316, -0.0437454, -0.0181764,
        0.000377467, 0.997513, 0.0704876, 0.0234806,
        0.0438279, -0.0704364, 0.996553, 0.122565,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999117, 0.00618585, -0.0415629, -0.0185552,
        -0.00280672, 0.996721, 0.0808731, 0.0245329,
        0.0419268, -0.0806851, 0.995858, 0.134509,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999251, 0.010021, -0.0373797, -0.020394,
        -0.00659844, 0.995868, 0.0905853, 0.0246993,
        0.038133, -0.0902708, 0.995187, 0.146222,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999371, 0.0121883, -0.0333047, -0.0208747,
        -0.00892676, 0.995305, 0.0963803, 0.0222097,
        0.0343231, -0.0960224, 0.994787, 0.157386,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99959, 0.00961161, -0.0269745, -0.0225797,
        -0.00709479, 0.995744, 0.0918949, 0.0221385,
        0.027743, -0.0916659, 0.995403, 0.168588,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999633, 0.00733784, -0.0260882, -0.0241657,
        -0.00510994, 0.996414, 0.084462, 0.0285891,
        0.0266144, -0.0842978, 0.996085, 0.18043,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999573, 0.00597872, -0.0286111, -0.024779,
        -0.00347664, 0.996228, 0.0867149, 0.0318192,
        0.0290216, -0.0865785, 0.995822, 0.192756,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999582, 0.00328094, -0.0287464, -0.0258026,
        -0.00072471, 0.996078, 0.0884861, 0.0334273,
        0.028924, -0.0884283, 0.995663, 0.205491,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999545, 0.00142751, -0.0301445, -0.0250534,
        0.00122198, 0.996148, 0.0876921, 0.0395994,
        0.0301535, -0.0876891, 0.995691, 0.2183,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999669, 0.00422681, -0.0253899, -0.0253594,
        -0.00188168, 0.995787, 0.0916876, 0.0419873,
        0.0256705, -0.0916095, 0.995464, 0.230988,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999786, 0.00681975, -0.0195495, -0.0256572,
        -0.00496749, 0.995628, 0.0932763, 0.0457138,
        0.0201002, -0.0931592, 0.995449, 0.243596,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999907, 0.00794488, -0.0110979, -0.0269755,
        -0.00683591, 0.995299, 0.0966169, 0.047554,
        0.0118134, -0.0965321, 0.99526, 0.255574,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999915, 0.0121027, -0.00493708, -0.0283011,
        -0.0115752, 0.995349, 0.09564, 0.0501272,
        0.00607163, -0.0955748, 0.995404, 0.267216,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999885, 0.0150932, -0.00162115, -0.0295151,
        -0.0148734, 0.995444, 0.094192, 0.0536427,
        0.00303544, -0.0941571, 0.995553, 0.278576,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999854, 0.0170969, -0.000857404, -0.0327324,
        -0.0169391, 0.995371, 0.0946113, 0.0561432,
        0.00247102, -0.094583, 0.995514, 0.289835,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999759, 0.0174862, -0.0132952, -0.0352916,
        -0.0161342, 0.995273, 0.0957713, 0.0594513,
        0.014907, -0.0955337, 0.995315, 0.301074,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999623, 0.0120191, -0.0247127, -0.0370757,
        -0.0095422, 0.99514, 0.0980113, 0.0628777,
        0.0257706, -0.0977386, 0.994879, 0.31217,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999325, 0.00750696, -0.0359786, -0.0378985,
        -0.0038312, 0.994863, 0.101165, 0.0659766,
        0.0365532, -0.100959, 0.994219, 0.322609,
        0, 0, 0, 1;

    ///////////////////////////////////////////
    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999218, 0.00263672, -0.0394559, -0.0385721,
        0.00136685, 0.994876, 0.1011, 0.0675941,
        0.0395203, -0.101075, 0.994094, 0.332305,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99917, 0.00219765, -0.0406725, -0.0379906,
        0.00173027, 0.995353, 0.0962878, 0.0737919,
        0.0406951, -0.0962783, 0.994522, 0.341691,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999221, -0.00083337, -0.0394467, -0.0388021,
        0.00476131, 0.995018, 0.0995871, 0.0763239,
        0.0391672, -0.0996975, 0.994247, 0.350638,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999292, -0.000511404, -0.0376108, -0.0387157,
        0.00424499, 0.995054, 0.0992565, 0.0818187,
        0.037374, -0.099346, 0.994351, 0.359382,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.999247, -0.00145, -0.038772, -0.0384784,
        0.00561561, 0.994185, 0.107547, 0.0865023,
        0.0383906, -0.107684, 0.993444, 0.368285,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99919, -0.00776856, -0.0394905, -0.0381415,
        0.0125291, 0.99248, 0.121771, 0.0892652,
        0.0382475, -0.122167, 0.991773, 0.376358,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.998839, -0.0164664, -0.0452749, -0.0403097,
        0.0227734, 0.989529, 0.142529, 0.0887485,
        0.0424539, -0.143394, 0.988755, 0.383166,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.998024, -0.0175865, -0.0603211, -0.0422629,
        0.0269605, 0.987023, 0.158301, 0.085191,
        0.0567543, -0.159614, 0.985547, 0.387593,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.997488, -0.0163879, -0.0689076, -0.0400306,
        0.0273081, 0.986632, 0.160659, 0.0854955,
        0.0653536, -0.162138, 0.984602, 0.390166,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.997532, -0.0191163, -0.0675622, -0.0387457,
        0.0303211, 0.985161, 0.168934, 0.0833311,
        0.0633303, -0.170566, 0.983309, 0.391008,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99756, -0.0232771, -0.0658126, -0.0396275,
        0.0345272, 0.9839, 0.175355, 0.0784991,
        0.0606714, -0.177199, 0.982303, 0.389852,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.997266, -0.021894, -0.0705811, -0.040002,
        0.034068, 0.983765, 0.176199, 0.0755142,
        0.0655776, -0.178121, 0.981821, 0.386711,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.997016, -0.0185577, -0.0749253, -0.0402658,
        0.0316286, 0.98366, 0.17724, 0.0707784,
        0.0704119, -0.179081, 0.981312, 0.382002,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.996939, -0.0142089, -0.076885, -0.0408578,
        0.0274725, 0.984306, 0.174319, 0.065278,
        0.0732016, -0.175898, 0.981683, 0.375335,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.996828, -0.0107155, -0.0788623, -0.0408441,
        0.0238734, 0.985523, 0.167853, 0.0595021,
        0.075922, -0.169203, 0.982653, 0.367276,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.996519, -0.0115473, -0.0825707, -0.04061,
        0.0246434, 0.986907, 0.159397, 0.0555861,
        0.0796491, -0.160877, 0.983755, 0.358087,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.995944, -0.0147179, -0.0887705, -0.0408885,
        0.0284589, 0.987414, 0.155578, 0.0501558,
        0.0853635, -0.157473, 0.983827, 0.348207,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.995204, -0.0154884, -0.0965941, -0.0403651,
        0.0299336, 0.988242, 0.149944, 0.0452793,
        0.093136, -0.152116, 0.983965, 0.337567,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.994708, -0.0153701, -0.101592, -0.0391879,
        0.030019, 0.989082, 0.144282, 0.0408364,
        0.0982657, -0.146567, 0.984308, 0.326267,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.994513, -0.0150301, -0.103532, -0.037756,
        0.0295096, 0.989743, 0.13978, 0.0360097,
        0.100369, -0.142068, 0.984755, 0.314498,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99462, -0.013158, -0.102758, -0.0368328,
        0.0271322, 0.990364, 0.135804, 0.0296793,
        0.0999809, -0.137862, 0.985392, 0.302012,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.994718, -0.0067043, -0.102427, -0.0354878,
        0.0199371, 0.99148, 0.128722, 0.0269947,
        0.100692, -0.130084, 0.986377, 0.288802,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99488, 0.000133946, -0.101067, -0.0340111,
        0.0128993, 0.991653, 0.128292, 0.0246807,
        0.10024, -0.128938, 0.986573, 0.276048,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.994793, 0.0060772, -0.101743, -0.0340748,
        0.00773051, 0.990847, 0.134769, 0.0221934,
        0.101631, -0.134854, 0.98564, 0.263529,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.994329, 0.0107082, -0.105813, -0.0360279,
        0.00531046, 0.988678, 0.149957, 0.0125956,
        0.10622, -0.149668, 0.983014, 0.250653,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.993582, 0.013471, -0.112312, -0.0363013,
        0.00367611, 0.988514, 0.151086, 0.00476259,
        0.113057, -0.150529, 0.98212, 0.236787,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.992921, 0.0156526, -0.11774, -0.0352797,
        0.00175654, 0.989235, 0.146324, 0.00128367,
        0.118763, -0.145495, 0.982205, 0.222258,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.992566, 0.0173375, -0.120474, -0.0345205,
        0.000115884, 0.989668, 0.143379, -0.00457096,
        0.121715, -0.142327, 0.982308, 0.207701,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.992572, 0.0190162, -0.120168, -0.031961,
        -0.00243568, 0.990617, 0.136644, -0.0114871,
        0.121639, -0.135336, 0.983305, 0.192957,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.993183, 0.0217036, -0.114529, -0.0295021,
        -0.00761921, 0.9925, 0.122009, -0.0143089,
        0.116318, -0.120305, 0.985899, 0.177862,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.993883, 0.0269068, -0.107111, -0.0257021,
        -0.0147588, 0.993527, 0.112632, -0.0158047,
        0.109448, -0.110362, 0.987847, 0.16343,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.995095, 0.0283374, -0.0947836, -0.024243,
        -0.0180365, 0.994007, 0.10782, -0.0223213,
        0.0972709, -0.105581, 0.989642, 0.149542,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.996137, 0.0272717, -0.0834825, -0.0242938,
        -0.0190197, 0.994996, 0.098093, -0.0267727,
        0.0857398, -0.0961261, 0.99167, 0.135799,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.99655, 0.0276002, -0.078287, -0.0234021,
        -0.0205443, 0.995772, 0.0895434, -0.0302442,
        0.0804273, -0.087626, 0.992902, 0.122264,
        0, 0, 0, 1;

    vladPoses.push_back(temp);
    temp = Matrix4f::Zero();
    temp << 0.996908, 0.0267689, -0.0738876, -0.0222569,
        -0.0207068, 0.996448, 0.0816246, -0.0347516,
        0.0758101, -0.0798422, 0.993921, 0.109025,
        0, 0, 0, 1;
    vladPoses.push_back(temp);
    // const std::string filenameIn = std::string("/home/marc/Projects/3DMotion-Scanning/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/");
    const std::string filenameIn = std::string("/rhome/mbenedi/datasets/rgbd_dataset_freiburg1_xyz/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg3_teddy/");

    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg2_rpy/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg3_cabinet/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg2_flowerbouquet_brownbackground/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg2_coke/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg1_plant/");
    // const std::string filenameIn = std::string("/home/antares/kyildiri/stuff/rgbd_dataset_freiburg1_xyz/");

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

    int it = 0;
    Matrix4f workingPose;
    std::cout << vladPoses.size() << std::endl;
    while (sensor.processNextFrame())
    {

        // std::vector<Vector3f> vertices;
        // cv::Mat volume;
        // volume.setTo(0);
        // model.getGPUGrid().download(volume);

        // for (int i = 1; i < 1023; i++)
        // {
        //     for (int j = 1; j < 1023; j++)
        //     {
        //         for (int k = 1; k < 1023; k++)
        //         {
        //             int ind = (i * 1024 + j) * 1024 + k;
        //             assert(ind >= 0);

        //             int indFront = (i * 1024 + j) * 1024 + k + 1;
        //             int indUp = ((i+1) * 1024 + j) * 1024 + k ;
        //             int indRight = (i * 1024 + j+1) * 1024 + k;
        //             float value = volume.at<cv::Vec2f>(ind)[0];
        //             float valueFront = volume.at<cv::Vec2f>(indFront)[0];
        //             float valueUp = volume.at<cv::Vec2f>(indUp)[0];
        //             float valueRight = volume.at<cv::Vec2f>(indRight)[0];

        //             if ((value * valueFront < 0  || value * valueUp < 0 || value * valueRight < 0) && value != 0)
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
        // pcd.writeMesh("tsdf_" + std::to_string(it) + ".off");

        // for (int level = 2; level >= 0; level--)
        // {
        //     bool validPose = Wrapper::poseEstimation(sensor, currentCameraToWorld, cameraParams,
        //                             model.getSurfacePoints(level), model.getSurfaceNormals(level), level);
        //     if(validPose) {
        //         workingPose = currentCameraToWorld;
        //     } else {
        //         // currentCameraToWorld = workingPose;
        //         // continue;
        //         return 0;
        //     }
        //     std::cout << "Level: " << level << std::endl;
        //     // return 0;
        //  }

        currentCameraToWorld = vladPoses[it];
        std::cout << currentCameraToWorld << std::endl;
        Wrapper::updateReconstruction(model, cameraParams, sensor.getDepth(), currentCameraToWorld.inverse());
        // PointCloud anan(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(),
        //                          sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
        // anan.writeMesh("SENSOR_" + std::to_string(it) + ".off");
        it++;
        if (it > 62)
        {
            return 0;
        }
        for (int level = 2; level >= 0; level--)
        {
            Wrapper::rayCast(model, cameraParams, currentCameraToWorld, level);
        }
        // if (it % 1 == 0)
        // {
        //     SimpleMesh currentDepthMesh{sensor, currentCameraToWorld.inverse(), 0.1f};
        //     SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraToWorld.inverse(), 0.0015f);
        //     SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

        //     std::stringstream ss;
        //     ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
        //     std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
        //     if (!resultingMesh.writeMesh(ss.str()))
        //     {
        //         std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        //         return -1;
        //     }
        // }
    }
    return 0;
}