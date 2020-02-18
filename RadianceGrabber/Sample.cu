#include "Sample.cuh"
#include "DataTypes.cuh"
#include <cmath>
#include <algorithm>

using namespace RadGrabber;

__constant__ __host__ __device__ const double Inv2Pi = 0.15915494309189533577;
__constant__ __host__ __device__ const double Inv4Pi = 0.07957747154594766788;
__constant__ __host__ __device__ const double Pi = 3.14159265358979323846;
__constant__ __host__ __device__ const double PiOver2 = 1.57079632679489661923;
__constant__ __host__ __device__ const double PiOver4 = 0.78539816339744830961;
__constant__ __host__ __device__ const double Sqrt2 = 1.41421356237309504880;

__host__ __device__ Vector3f UniformSampleHemisphere(const Vector2f &u) {
	float z = u[0];
	float r = std::sqrt(std::max((float)0, (float)1. - z * z));
	float phi = 2 * Pi * u[1];
	return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
}

__host__ __device__ float UniformHemispherePdf() { return Inv2Pi; }

__host__ __device__ Vector3f UniformSampleSphere(const Vector2f &u) {
	float z = 1 - 2 * u[0];
	float r = std::sqrt(std::max((float)0, (float)1 - z * z));
	float phi = 2 * Pi * u[1];
	return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
}

__host__ __device__ float UniformSpherePdf() { return Inv4Pi; }

__host__ __device__ Vector2f UniformSampleDisk(const Vector2f &u) {
	float r = std::sqrt(u[0]);
	float theta = 2 * Pi * u[1];
	return Vector2f(r * std::cos(theta), r * std::sin(theta));
}

__host__ __device__ Vector2f ConcentricSampleDisk(const Vector2f &u) {
	// Map uniform random numbers to $[-1,1]^2$
	Vector2f uOffset = 2.f * u - Vector2f(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0) return Vector2f(0, 0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = PiOver4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
	}
	return r * Vector2f(std::cos(theta), std::sin(theta));
}

__host__ __device__ float UniformConePdf(float cosThetaMax) {
	return 1 / (2 * Pi * (1 - cosThetaMax));
}

__host__ __device__ Vector3f UniformSampleCone(const Vector2f &u, float cosThetaMax) {
	float cosTheta = ((float)1 - u[0]) + u[0] * cosThetaMax;
	float sinTheta = std::sqrt((float)1 - cosTheta * cosTheta);
	float phi = u[1] * 2 * Pi;
	return Vector3f(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta,
		cosTheta);
}