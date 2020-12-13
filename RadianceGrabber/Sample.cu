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

	__host__ __device__ Vector3f UniformSampleHemisphere(const Vector2f &u) {
		float z = u[0];
		float r = std::sqrt(fmax((float)0, (float)1. - z * z));
		float phi = 2.f * Pi * u[1];
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
		float phi = 2.f * Pi * u[1];

		return b1 * r * std::cos(phi) + n * r * std::sin(phi) + b2 * z;
	}

	__host__ __device__ float UniformHemispherePdf() { return (float)Inv2Pi; }

	__host__ __device__ Vector3f UniformSampleSphere(const Vector2f &u) {
		float z = 1 - 2 * u[0];
		float r = std::sqrt(fmax((float)0, (float)1 - z * z));
		float phi = 2.f * Pi * u[1];
		return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
	}

	__host__ __device__ float UniformSpherePdf() { return (float)Inv4Pi; }

	__host__ __device__ Vector2f UniformSampleDisk(const Vector2f &u) {
		float r = std::sqrt(u[0]);
		float theta = 2.f * Pi * u[1];
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
			theta = float(PiOver4 * (uOffset.y / uOffset.x));
		}
		else {
			r = uOffset.y;
			theta = float(PiOver2 - PiOver4 * (uOffset.x / uOffset.y));
		}
		return r * Vector2f(std::cos(theta), std::sin(theta));
	}

	__host__ __device__ float UniformConePdf(float cosThetaMax) {
		return float(1 / (2 * Pi * (1 - cosThetaMax)));
	}

	__host__ __device__ Vector3f UniformSampleCone(const Vector2f &u, float cosThetaMax) {
		float cosTheta = ((float)1 - u.x) + u.x * cosThetaMax;
		float sinTheta = std::sqrt((float)1 - cosTheta * cosTheta);
		float phi = u.y * 2.f * Pi;
		return Vector3f(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta,
			cosTheta);
	}

	__host__ __device__ Vector3f UniformSampleConeInFrame(const Vector3f& n, const Vector2f &u, float cosThetaMax) {
		float cosTheta = ((float)1 - u.x) + u.x * cosThetaMax;
		float sinTheta = std::sqrt((float)1 - cosTheta * cosTheta);
		float phi = u.y * 2 * Pi;

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


	//__host__ __device__ inline float Saturate(float value) { return fmin(fmax(value, 0), 1); }

	//__host__ __device__ inline float SchlickWeight(float cosTheta) {
	//	float m = Saturate(1 - cosTheta);
	//	return (m * m) * (m * m) * m;
	//}
	//__host__ __device__ inline float FresnelSchlickApprox(float R0, float cosTheta) {
	//	return R0 + (1.0f - R0) * SchlickWeight(cosTheta);
	//}
	//__host__ __device__ inline float RoughnessToAlphaTrowbridgeReitz(float roughness) {
	//	roughness = fmax(roughness, (float)1e-3);
	//	float x = std::log(roughness);
	//	return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
	//		0.000640711f * x * x * x * x;
	//}

	//inline float GGX_G1(const Vector3f& n, const Vector3f& v, const float NdotV, const float alpha)
	//{
	//	return 2 * NdotV / (NdotV + sqrtf(alpha*alpha + (1 - alpha*alpha)*NdotV*NdotV));
	//}
	//inline float GGX_D(const Vector3f& n, const Vector3f& v, const float NdotV, const float alpha)
	//{
	//	float denom = NdotV * NdotV * (alpha*alpha - 1) + 1;
	//	return alpha * alpha / (PI * denom * denom);
	//}

	//__device__ void MicrofacetInteraction(
	//	const MaterialChunk* mat, float metallic, float smoothness,
	//	IN const Texture2DChunk* textureBuffer, IN SurfaceIntersection* isect, IN curandState* state,
	//	INOUT Vector4f& attenuPerComp, INOUT Ray& ray) 
	//{
	//	Vector2f u2(curand_uniform(state), curand_uniform(state));
	//	float	ndotv = Dot(isect->normal, -ray.direction),
	//			fresnel = FresnelSchlickApprox(mat->URPLit.metallic, ndotv),
	//			alpha = RoughnessToAlphaTrowbridgeReitz(1 - smoothness);

	//	Vector3f wo;
	//	if (u2[0] < fresnel)
	//	{
	//		Vector3f microNormal;
	//		wo = Reflect(ray.direction, microNormal);
	//	}
	//	else
	//	{
	//		wo = UniformSampleHemisphereInFrame(isect->normal, u2);
	//	}

	//	Vector3f wh = (wo - ray.direction).normalized();
	//	float D, G;



	//	ray.direction = wo;
	//}
}