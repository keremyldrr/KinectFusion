#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <flann/flann.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
// #include "ProcrustesAligner.h"

/**
 * ICP optimizer - Abstract Base Class, using Ceres for optimization.
 */
class ICPOptimizer
{
public:
	ICPOptimizer() : m_bUsePointToPlaneConstraints{false},
					 m_nIterations{20},
					 m_nearestNeighborSearch{std::make_unique<NearestNeighborSearchFlann>()}
	{
	}

	void setMatchingMaxDistance(float maxDistance)
	{
		m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
	}

	void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints)
	{
		m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
	}

	void setNbOfIterations(unsigned nIterations)
	{
		m_nIterations = nIterations;
	}

	virtual Matrix4f estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f initialPose = Matrix4f::Identity()) = 0;

protected:
	bool m_bUsePointToPlaneConstraints;
	unsigned m_nIterations;
	std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

	std::vector<Vector3f> transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose)
	{
		std::vector<Vector3f> transformedPoints;
		transformedPoints.reserve(sourcePoints.size());

		const auto rotation = pose.block(0, 0, 3, 3);
		const auto translation = pose.block(0, 3, 3, 1);

		for (const auto &point : sourcePoints)
		{
			transformedPoints.push_back(rotation * point + translation);
		}

		return transformedPoints;
	}

	std::vector<Vector3f> transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose)
	{
		std::vector<Vector3f> transformedNormals;
		transformedNormals.reserve(sourceNormals.size());

		const auto rotation = pose.block(0, 0, 3, 3);

		for (const auto &normal : sourceNormals)
		{
			transformedNormals.push_back(rotation.inverse().transpose() * normal);
		}

		return transformedNormals;
	}

	void pruneCorrespondences(const std::vector<Vector3f> &sourceNormals, const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches)
	{
		const unsigned nPoints = sourceNormals.size();

		for (unsigned i = 0; i < nPoints; i++)
		{
			Match &match = matches[i];
			if (match.idx >= 0)
			{
				const auto &sourceNormal = sourceNormals[i];
				const auto &targetNormal = targetNormals[match.idx];

				// TODO: [DONE] If the distances are shitty
				auto ns = sourceNormal.normalized();
				auto tg = targetNormal.normalized();

				auto check = acos(ns.dot(tg)) * 180 / EIGEN_PI;
				if (check > 60)
				{
					matches[i].idx = -1;
					//matches[i].weight = -1;
				}
			}
		}
	}
};

/**
 * ICP optimizer - using Ceres for optimization.
 */
class LinearICPOptimizer : public ICPOptimizer
{
public:
	LinearICPOptimizer() {}

	 virtual Matrix4f estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f initialPose = Matrix4f::Identity()) override
	{
		// Build the index of the FLANN tree (for fast nearest neighbor lookup).
		// std::vector<Vector3f> sourcePoints;
		// std::vector<Vector3f> sourceNormals;
		// std::vector<Vector3f> targetPoints;
		// std::vector<Vector3f> targetNormals;
		// cv::Mat hostVertexMap = sensor.getVertexMap();
		// cv::Mat hostNormalMap = sensor.getNormalMap();
		// for (int i = 0; i < cameraParams.depthImageHeight; i++)
		// {
		// 	for (int j = 0; j < cameraParams.depthImageWidth; j++)
		// 	{
		// 		{
		// 			Vector3f pnt;
		// 			pnt.x() = hostVertexMap.at<cv::Vec3f>(i, j)[0];
		// 			pnt.y() = hostVertexMap.at<cv::Vec3f>(i, j)[1];
		// 			pnt.z() = hostVertexMap.at<cv::Vec3f>(i, j)[2];
		// 			Vector3f normal;
		// 			normal.x() = hostNormalMap.at<cv::Vec3f>(i, j)[0];
		// 			normal.y() = hostNormalMap.at<cv::Vec3f>(i, j)[1];
		// 			normal.z() = hostNormalMap.at<cv::Vec3f>(i, j)[2];

		// 			targetPoints.push_back(pnt);
		// 			targetNormals.push_back(normal);
		// 			// printf("source %d %d --> target %d %d \n",i,j,x,y);
		// 		}
		// 	}
		// }

		// PointCloud target(targetPoints, targetNormals);
		// PointCloud source(sourcePoints, sourceNormals);
		m_nearestNeighborSearch->buildIndex(target.getPoints());

		// The initial estimate can be given as an argument.
		Matrix4f estimatedPose = initialPose;

		for (int i = 0; i < m_nIterations; ++i)
		{
			int cnt = 0;

			// Compute the matches.
			std::cout << "Matching points ..." << std::endl;
			clock_t begin = clock();

			auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
			auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

			auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
			pruneCorrespondences(transformedNormals, target.getNormals(), matches);

			clock_t end = clock();
			double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

			std::vector<Vector3f> sourcePoints;
			std::vector<Vector3f> targetPoints;

			// Add all matches to the sourcePoints and targetPoints vectors,
			// so that sourcePoints[i] matches targetPoints[i].
			for (int j = 0; j < transformedPoints.size(); j++)
			{
				const auto &match = matches[j];
				if (match.idx >= 0)
				{
					cnt++;
					sourcePoints.push_back(transformedPoints[j]);
					targetPoints.push_back(target.getPoints()[match.idx]);
				}
			}

			// Estimate the new pose
			// if (m_bUsePointToPlaneConstraints) {
			estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;
			// }
			// else {
			// 	estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
			// }

			std::cout << "Optimization iteration done."
					  << " " << cnt << std::endl;
		}

		return estimatedPose;
	}

private:
	// Matrix4f estimatePosePointToPoint(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints)
	// {
	// 	// ProcrustesAligner procrustAligner;
	// 	// Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

	// 	// return estimatedPose;
	// }

	Matrix4f estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals)
	{
		const unsigned nPoints = sourcePoints.size();

		// Build the system
		MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
		VectorXf b = VectorXf::Zero(4 * nPoints);

		for (unsigned i = 0; i < nPoints; i++)
		{
			const auto &s = sourcePoints[i];
			const auto &d = targetPoints[i];
			const auto &n = targetNormals[i];

			// TODO: [DONE] Add the point-to-plane constraints to the system
			//  1 point-to-plane row per point
			A(4 * i, 0) = n.z() * s.y() - n.y() * s.z();
			A(4 * i, 1) = n.x() * s.z() - n.z() * s.x();
			A(4 * i, 2) = n.y() * s.x() - n.x() * s.y();
			A(4 * i, 3) = n.x();
			A(4 * i, 4) = n.y();
			A(4 * i, 5) = n.z();
			b(4 * i) = n.x() * d.x() + n.y() * d.y() + n.z() * d.z() - n.x() * s.x() - n.y() * s.y() - n.z() * s.z();

			// TODO: [DONE] Add the point-to-point constraints to the system
			//  3 point-to-point rows per point (one per coordinate)
			A(4 * i + 1, 0) = 0.0f;
			A(4 * i + 1, 1) = s.z();
			A(4 * i + 1, 2) = -s.y();
			A(4 * i + 1, 3) = 1.0f;
			A(4 * i + 1, 4) = 0.0f;
			A(4 * i + 1, 5) = 0.0f;
			b(4 * i + 1) = d.x() - s.x();

			A(4 * i + 2, 0) = -s.z();
			A(4 * i + 2, 1) = 0.0f;
			A(4 * i + 2, 2) = s.x();
			A(4 * i + 2, 3) = 0.0f;
			A(4 * i + 2, 4) = 1.0f;
			A(4 * i + 2, 5) = 0.0f;
			b(4 * i + 2) = d.y() - s.y();

			A(4 * i + 3, 0) = s.y();
			A(4 * i + 3, 1) = -s.x();
			A(4 * i + 3, 2) = 0.0f;
			A(4 * i + 3, 3) = 0.0f;
			A(4 * i + 3, 4) = 0.0f;
			A(4 * i + 3, 5) = 1.0f;
			b(4 * i + 3) = d.z() - s.z();

			// TODO: [DONE] Optionally, apply a higher weight to point-to-plane correspondences
			float LAMBDA_plane = 1.0f;
			float LAMBDA_point = 0.1f;
			A(4 * i) *= LAMBDA_plane;
			b(4 * i) *= LAMBDA_plane;

			A(4 * i + 1) *= LAMBDA_point;
			b(4 * i + 1) *= LAMBDA_point;
			A(4 * i + 2) *= LAMBDA_point;
			b(4 * i + 2) *= LAMBDA_point;
			A(4 * i + 3) *= LAMBDA_point;
			b(4 * i + 3) *= LAMBDA_point;
		}

		// TODO: [DONE] Solve the system
		VectorXf x(6);

		JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
		const MatrixXf &E_i = svd.singularValues().asDiagonal().inverse();
		const MatrixXf &U_t = svd.matrixU().transpose();
		const MatrixXf &V = svd.matrixV();

		x = V * E_i * U_t * b;

		float alpha = x(0), beta = x(1), gamma = x(2);

		// Build the pose matrix
		Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
							AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
							AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

		Vector3f translation = x.tail(3);

		// TODO: [DONE] Build the pose matrix using the rotation and translation matrices
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block<3, 3>(0, 0) = rotation;
		estimatedPose.block<3, 1>(0, 3) = translation;

		return estimatedPose;
	}
};
