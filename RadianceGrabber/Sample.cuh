#include "DataTypes.cuh"

#pragma once
 
namespace RadGrabber
{
	__host__ __device__ Vector3f UniformSampleHemisphere(const Vector2f &u);
	__host__ __device__ Vector3f UniformSampleHemisphereInFrame(const Vector3f& n, const Vector2f &u);
	__host__ __device__ float UniformHemispherePdf();
	__host__ __device__ Vector3f UniformSampleSphere(const Vector2f &u);
	__host__ __device__ float UniformSpherePdf();
	__host__ __device__ Vector2f UniformSampleDisk(const Vector2f &u);
	__host__ __device__ Vector2f ConcentricSampleDisk(const Vector2f &u);
	__host__ __device__ float UniformConePdf(float cosThetaMax);
	__host__ __device__ Vector3f UniformSampleCone(const Vector2f &u, float cosThetaMax);
	__host__ __device__ Vector3f UniformSampleConeInFrame(const Vector3f& n, const Vector2f &u, float cosThetaMax);

	__host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, float& lastDistance, Vector3f& bc);
	__host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, Vector3f& bc);
}
