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

	__forceinline__ __host__ __device__ Vector3f CosineSampleHemisphere(const Vector2f &u) {
		Vector2f d = ConcentricSampleDisk(u);
		float z = sqrtf(fmax((float)0, 1 - d.x * d.x - d.y * d.y));
		return Vector3f(d.x, d.y, z);
	}

	inline float CosineHemispherePdf(float cosTheta) { return cosTheta * InvPi; }

	inline float BalanceHeuristic(int nf, float fPdf, int ng, float gPdf) {
		return (nf * fPdf) / (nf * fPdf + ng * gPdf);
	}

	inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
		float f = nf * fPdf, g = ng * gPdf;
		return (f * f) / (f * f + g * g);
	}

	__forceinline__ __host__ __device__ float gamma(int n) 
	{
		return (float)(n * MachineEpsilon) / (1 - n * MachineEpsilon);
	}

	__forceinline__ __host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, float& lastDistance, Vector3f& bc)
	{
		// PBRTv3:: Translate vertices based on ray origin
		Vector3f p0t = p0 - ray.origin;
		Vector3f p1t = p1 - ray.origin;
		Vector3f p2t = p2 - ray.origin;

		// PBRTv3:: Permute components of triangle vertices and ray direction
		int kz = MaxDimension(Abs(ray.direction));
		int kx = (kz + 1) % 3;
		int ky = (kx + 1) % 3;
		Vector3f d = Permute(ray.direction, kx, ky, kz);
		p0t = Permute(p0t, kx, ky, kz);
		p1t = Permute(p1t, kx, ky, kz);
		p2t = Permute(p2t, kx, ky, kz);

		// PBRTv3:: Apply shear transformation to translated vertex positions
		float Sx = -d.x / d.z;
		float Sy = -d.y / d.z;
		float Sz = 1.f / d.z;
		p0t.x += Sx * p0t.z;
		p0t.y += Sy * p0t.z;
		p1t.x += Sx * p1t.z;
		p1t.y += Sy * p1t.z;
		p2t.x += Sx * p2t.z;
		p2t.y += Sy * p2t.z;

		// PBRTv3:: Compute edge function coefficients _e0_, _e1_, and _e2_
		float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
		float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
		float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

		// PBRTv3:: Fall back to double precision test at triangle edges
		if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) {
			double p2txp1ty = (double)p2t.x * (double)p1t.y;
			double p2typ1tx = (double)p2t.y * (double)p1t.x;
			e0 = (float)(p2typ1tx - p2txp1ty);
			double p0txp2ty = (double)p0t.x * (double)p2t.y;
			double p0typ2tx = (double)p0t.y * (double)p2t.x;
			e1 = (float)(p0typ2tx - p0txp2ty);
			double p1txp0ty = (double)p1t.x * (double)p0t.y;
			double p1typ0tx = (double)p1t.y * (double)p0t.x;
			e2 = (float)(p1typ0tx - p1txp0ty);
		}

		// PBRTv3:: Perform triangle edge and determinant tests
		if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
			return false;
		float det = e0 + e1 + e2;
		if (det == 0)
			return false;

		// PBRTv3:: Compute scaled hit distance to triangle and test against ray $t$ range
		p0t.z *= Sz;
		p1t.z *= Sz;
		p2t.z *= Sz;
		float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
		if (det < 0 && (tScaled >= 0/* || tScaled < ray.tMax * det*/))
			return false;
		else if (det > 0 && (tScaled <= 0/* || tScaled > ray.tMax * det*/))
			return false;

		// PBRTv3:: Compute barycentric coordinates and $t$ value for triangle intersection
		float invDet = 1 / det;
		bc.x = e0 * invDet;
		bc.y = e1 * invDet;
		bc.z = e2 * invDet;
		float t = tScaled * invDet;

		// distance check
		if (t > lastDistance)
			return false;

		// PBRTv3:: Ensure that computed triangle $t$ is conservatively greater than zero

		// PBRTv3:: Compute $\delta_z$ term for triangle $t$ error bounds
		float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
		float deltaZ = gamma(3) * maxZt;

		// PBRTv3:: Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
		float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
		float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
		float deltaX = gamma(5) * (maxXt + maxZt);
		float deltaY = gamma(5) * (maxYt + maxZt);

		// PBRTv3:: Compute $\delta_e$ term for triangle $t$ error bounds
		float deltaE =
			2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

		// PBRTv3:: Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
		float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
		float deltaT = 3 *
			(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
			std::abs(invDet);

		if (t > deltaT)
		{
			lastDistance = t;
			return true;
		}
		else
			return false;
	}
	__forceinline__ __host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, Vector3f& bc)
	{
		// PBRTv3:: Translate vertices based on ray origin
		Vector3f p0t = p0 - ray.origin;
		Vector3f p1t = p1 - ray.origin;
		Vector3f p2t = p2 - ray.origin;

		// PBRTv3:: Permute components of triangle vertices and ray direction
		int kz = MaxDimension(Abs(ray.direction));
		int kx = kz + 1;
		if (kx == 3) kx = 0;
		int ky = kx + 1;
		if (ky == 3) ky = 0;
		Vector3f d = Permute(ray.direction, kx, ky, kz);
		p0t = Permute(p0t, kx, ky, kz);
		p1t = Permute(p1t, kx, ky, kz);
		p2t = Permute(p2t, kx, ky, kz);

		// PBRTv3:: Apply shear transformation to translated vertex positions
		float Sx = -d.x / d.z;
		float Sy = -d.y / d.z;
		float Sz = 1.f / d.z;
		p0t.x += Sx * p0t.z;
		p0t.y += Sy * p0t.z;
		p1t.x += Sx * p1t.z;
		p1t.y += Sy * p1t.z;
		p2t.x += Sx * p2t.z;
		p2t.y += Sy * p2t.z;

		// PBRTv3:: Compute edge function coefficients _e0_, _e1_, and _e2_
		float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
		float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
		float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

		// PBRTv3:: Fall back to double precision test at triangle edges
		if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) {
			double p2txp1ty = (double)p2t.x * (double)p1t.y;
			double p2typ1tx = (double)p2t.y * (double)p1t.x;
			e0 = (float)(p2typ1tx - p2txp1ty);
			double p0txp2ty = (double)p0t.x * (double)p2t.y;
			double p0typ2tx = (double)p0t.y * (double)p2t.x;
			e1 = (float)(p0typ2tx - p0txp2ty);
			double p1txp0ty = (double)p1t.x * (double)p0t.y;
			double p1typ0tx = (double)p1t.y * (double)p0t.x;
			e2 = (float)(p1typ0tx - p1txp0ty);
		}

		// PBRTv3:: Perform triangle edge and determinant tests
		if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
			return false;
		float det = e0 + e1 + e2;
		if (det == 0)
			return false;

		// PBRTv3:: Compute scaled hit distance to triangle and test against ray $t$ range
		p0t.z *= Sz;
		p1t.z *= Sz;
		p2t.z *= Sz;
		float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
		if (det < 0 && (tScaled >= 0/* || tScaled < ray.tMax * det*/))
			return false;
		else if (det > 0 && (tScaled <= 0/* || tScaled > ray.tMax * det*/))
			return false;

		// PBRTv3:: Compute barycentric coordinates and $t$ value for triangle intersection
		float invDet = 1 / det;
		bc.x = e0 * invDet;
		bc.y = e1 * invDet;
		bc.z = e2 * invDet;
		float t = tScaled * invDet;

		// PBRTv3:: Ensure that computed triangle $t$ is conservatively greater than zero

		// PBRTv3:: Compute $\delta_z$ term for triangle $t$ error bounds
		float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
		float deltaZ = gamma(3) * maxZt;

		// PBRTv3:: Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
		float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
		float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
		float deltaX = gamma(5) * (maxXt + maxZt);
		float deltaY = gamma(5) * (maxYt + maxZt);

		// PBRTv3:: Compute $\delta_e$ term for triangle $t$ error bounds
		float deltaE =
			2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

		// PBRTv3:: Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
		float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
		float deltaT = 3 *
			(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
			std::abs(invDet);

		return t > deltaT;
	}
	__forceinline__ __host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, Vector3f& bc, float distance)
	{
		// PBRTv3:: Translate vertices based on ray origin
		Vector3f p0t = p0 - ray.origin;
		Vector3f p1t = p1 - ray.origin;
		Vector3f p2t = p2 - ray.origin;

		// PBRTv3:: Permute components of triangle vertices and ray direction
		int kz = MaxDimension(Abs(ray.direction));
		int kx = kz + 1;
		if (kx == 3) kx = 0;
		int ky = kx + 1;
		if (ky == 3) ky = 0;
		Vector3f d = Permute(ray.direction, kx, ky, kz);
		p0t = Permute(p0t, kx, ky, kz);
		p1t = Permute(p1t, kx, ky, kz);
		p2t = Permute(p2t, kx, ky, kz);

		// PBRTv3:: Apply shear transformation to translated vertex positions
		float Sx = -d.x / d.z;
		float Sy = -d.y / d.z;
		float Sz = 1.f / d.z;
		p0t.x += Sx * p0t.z;
		p0t.y += Sy * p0t.z;
		p1t.x += Sx * p1t.z;
		p1t.y += Sy * p1t.z;
		p2t.x += Sx * p2t.z;
		p2t.y += Sy * p2t.z;

		// PBRTv3:: Compute edge function coefficients _e0_, _e1_, and _e2_
		float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
		float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
		float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

		// PBRTv3:: Fall back to double precision test at triangle edges
		if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) {
			double p2txp1ty = (double)p2t.x * (double)p1t.y;
			double p2typ1tx = (double)p2t.y * (double)p1t.x;
			e0 = (float)(p2typ1tx - p2txp1ty);
			double p0txp2ty = (double)p0t.x * (double)p2t.y;
			double p0typ2tx = (double)p0t.y * (double)p2t.x;
			e1 = (float)(p0typ2tx - p0txp2ty);
			double p1txp0ty = (double)p1t.x * (double)p0t.y;
			double p1typ0tx = (double)p1t.y * (double)p0t.x;
			e2 = (float)(p1typ0tx - p1txp0ty);
		}

		// PBRTv3:: Perform triangle edge and determinant tests
		if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
			return false;
		float det = e0 + e1 + e2;
		if (det == 0)
			return false;

		// PBRTv3:: Compute scaled hit distance to triangle and test against ray $t$ range
		p0t.z *= Sz;
		p1t.z *= Sz;
		p2t.z *= Sz;
		float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
		if (det < 0 && (tScaled >= 0/* || tScaled < ray.tMax * det*/))
			return false;
		else if (det > 0 && (tScaled <= 0/* || tScaled > ray.tMax * det*/))
			return false;

		// PBRTv3:: Compute barycentric coordinates and $t$ value for triangle intersection
		float invDet = 1 / det;
		bc.x = e0 * invDet;
		bc.y = e1 * invDet;
		bc.z = e2 * invDet;
		float t = tScaled * invDet;

		// PBRTv3:: Ensure that computed triangle $t$ is conservatively greater than zero

		// PBRTv3:: Compute $\delta_z$ term for triangle $t$ error bounds
		float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
		float deltaZ = gamma(3) * maxZt;

		// PBRTv3:: Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
		float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
		float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
		float deltaX = gamma(5) * (maxXt + maxZt);
		float deltaY = gamma(5) * (maxYt + maxZt);

		// PBRTv3:: Compute $\delta_e$ term for triangle $t$ error bounds
		float deltaE =
			2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

		// PBRTv3:: Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
		float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
		float deltaT = 3 *
			(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
			std::abs(invDet);

		return t > deltaT;
	}
}
