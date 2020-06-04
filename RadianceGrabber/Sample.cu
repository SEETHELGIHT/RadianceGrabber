#include "Sample.cuh"
#include "DataTypes.cuh"
#include <cmath>
#include <algorithm>
#include "Marshal.cuh"


namespace RadGrabber
{
	/*
		PBRTv3 codes
	*/

	__constant__ __device__ const double Inv2Pi = 0.15915494309189533577;
	__constant__ __device__ const double Inv4Pi = 0.07957747154594766788;
	__constant__ __device__ const double Pi = 3.14159265358979323846;
	__constant__ __device__ const double PiOver2 = 1.57079632679489661923;
	__constant__ __device__ const double PiOver4 = 0.78539816339744830961;
	__constant__ __device__ const double Sqrt2 = 1.41421356237309504880;

	__host__ __device__ Vector3f UniformSampleHemisphere(const Vector2f &u) {
		float z = u[0];
		float r = std::sqrt(fmax((float)0, (float)1. - z * z));
		float phi = 2 * Pi * u[1];
		return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
	}

	__host__ __device__ Vector3f UniformSampleHemisphereInFrame(const Vector3f& n, const Vector2f &u) {
		Vector3f b1, b2;
		if (abs(n.x) < SQRT_OF_ONE_THIRD)
			b1 = Vector3f(1, 0, 0);
		else if (abs(n.y) < SQRT_OF_ONE_THIRD)
			b1 = Vector3f(0, 1, 0);
		else
			b1 = Vector3f(0, 0, 1);
		b2 = Cross(n, b1).normalized();
		b1 = Cross(n, b2).normalized();

		float z = u[0];
		float r = std::sqrt(fmax((float)0, (float)1. - z * z));
		float phi = 2 * Pi * u[1];

		return b1 * r * std::cos(phi) + n * r * std::sin(phi) + b2 * z;
	}

	__host__ __device__ float UniformHemispherePdf() { return Inv2Pi; }

	__host__ __device__ Vector3f UniformSampleSphere(const Vector2f &u) {
		float z = 1 - 2 * u[0];
		float r = std::sqrt(fmax((float)0, (float)1 - z * z));
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
		float cosTheta = ((float)1 - u.x) + u.x * cosThetaMax;
		float sinTheta = std::sqrt((float)1 - cosTheta * cosTheta);
		float phi = u.y * 2 * Pi;
		return Vector3f(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta,
			cosTheta);
	}

	__host__ __device__ Vector3f UniformSampleConeInFrame(const Vector3f& n, const Vector2f &u, float cosThetaMax) {
		float cosTheta = ((float)1 - u.x) + u.x * cosThetaMax;
		float sinTheta = std::sqrt((float)1 - cosTheta * cosTheta);
		float phi = u.y * 2 * PI;

		Vector3f b1, b2;
		if (abs(n.x) < SQRT_OF_ONE_THIRD)
			b1 = Vector3f(1, 0, 0);
		else if (abs(n.y) < SQRT_OF_ONE_THIRD)
			b1 = Vector3f(0, 1, 0);
		else
			b1 = Vector3f(0, 0, 1);
		b2 = Cross(n, b1).normalized();
		b1 = Cross(n, b2).normalized();

		return b1 * std::cos(phi) * sinTheta + n * std::sin(phi) * sinTheta + b2 * cosTheta;
	}

	__forceinline__ __host__ __device__ float gamma(int n) {
		return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
	}

	__host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, float& lastDistance, Vector3f& bc)
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
	__host__ __device__ bool IntersectRayAndTriangle(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, Vector3f& bc)
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

	__host__ __device__ inline float Saturate(float value) { return fmin(fmax(value, 0), 1); }
	__host__ __device__ inline float Lerp(float t, float v1, float v2) { return (1 - t) * v1 + t * v2; }

	__host__ __device__ inline float SchlickWeight(float cosTheta) {
		float m = Saturate(1 - cosTheta);
		return (m * m) * (m * m) * m;
	}
	__host__ __device__ inline float FresnelSchlickApprox(float R0, float cosTheta) {
		return R0 + (1.0f - R0) * SchlickWeight(cosTheta);
	}
	__host__ __device__ inline float RoughnessToAlphaTrowbridgeReitz(float roughness) {
		roughness = fmax(roughness, (float)1e-3);
		float x = std::log(roughness);
		return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
			0.000640711f * x * x * x * x;
	}

	inline float GGX_G1(const Vector3f& n, const Vector3f& v, const float NdotV, const float alpha)
	{
		return 2 * NdotV / (NdotV + sqrtf(alpha*alpha + (1 - alpha*alpha)*NdotV*NdotV));
	}
	inline float GGX_D(const Vector3f& n, const Vector3f& v, const float NdotV, const float alpha)
	{
		float denom = NdotV * NdotV * (alpha*alpha - 1) + 1;
		return alpha * alpha / (PI * denom * denom);
	}

	__device__ void MicrofacetInteraction(
		const MaterialChunk* mat, float metallic, float smoothness,
		IN const Texture2DChunk* textureBuffer, IN SurfaceIntersection* isect, IN curandState* state,
		INOUT Vector4f& attenuPerComp, INOUT Ray& ray) 
	{
		Vector2f u2(curand_uniform(state), curand_uniform(state));
		float	ndotv = Dot(isect->normal, -ray.direction),
				fresnel = FresnelSchlickApprox(mat->URPLit.metallic, ndotv),
				alpha = RoughnessToAlphaTrowbridgeReitz(1 - smoothness);

		Vector3f wo;
		if (u2[0] < fresnel)
		{
			Vector3f microNormal;
			wo = Reflect(ray.direction, microNormal);
		}
		else
		{
			wo = UniformSampleHemisphereInFrame(isect->normal, u2);
		}

		Vector3f wh = (wo - ray.direction).normalized();
		float D, G;



		ray.direction = wo;
	}
}