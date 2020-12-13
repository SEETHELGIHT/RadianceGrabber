#include "Microfacet.h"
#include "Sample.cuh"
/* 
	PBRTv3 functions
*/
namespace RadGrabber
{

	__host__ __device__ Vector3f MicrofacetBRDF(const Vector3f &wo, const Vector3f &wi, float alpha, float lambdaWo, const Vector3f& F, float D) 
	{
		float cosThetaO = AbsCosThetaInLocal(wo), cosThetaI = AbsCosThetaInLocal(wi);
		Vector3f wh = wi + wo;
		// Handle degenerate cases for microfacet reflection
		if (cosThetaI == 0 || cosThetaO == 0) return 0.f;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return 0.f;
		wh = wh.normalized();
		float	lambdaWi = TrowbridgeReitzLambda(wi, alpha, alpha);
		return /*R * */D * MicrofacetG(lambdaWo, lambdaWi) * F / (4 * cosThetaI * cosThetaO);
	}

	__host__ __device__ float PDFOfMicrofacetBRDF(const Vector3f &wo, const Vector3f &wi, float D, float G1)
	{
		if (!SameHemisphere(wo, wi)) return 0;
		Vector3f wh = (wo + wi).normalized();
		return MicrofacetPdf(wo, wh, D, G1) / (4 * Dot(wo, wh));
	}

	__host__ __device__ Vector3f SampleMicrofacetBRDF(const Vector3f &wo, Vector3f *wi, const Vector3f &u, float *pdf, float metallic, const Vector3f& R0, float alpha, float upperIOR, float surfaceIOR)
	{
		// Sample microfacet orientation $\wh$ and reflected direction $\wi$
		if (wo.z == 0) return 0.;
		Vector3f wh = TrowbridgeReitzSampleWh(wo, u, alpha, alpha);
		*wi = Reflect(wo, wh);
		if (!SameHemisphere(wo, *wi)) return 0.f;

		float lambdaWo = TrowbridgeReitzLambda(wo, alpha, alpha);
		float D = TrowbridgeReitzDistribution(wh, alpha, alpha);
		float G1 = MicrofacetG1(lambdaWo);

		Vector3f F = DisneyFresnelWithSurfaceIOR(metallic, R0, upperIOR, surfaceIOR, Dot(*wi, wh));

		// Pick specular=microfacet BRDF vs diffuse=lambertian by probability
		if (u.z < Luminance(F))
		{
			// Compute PDF of _wi_ for microfacet reflection
			*pdf = MicrofacetPdf(wo, wh, D, G1) / (4 * Dot(wo, wh));
			return MicrofacetBRDF(wo, *wi, alpha, lambdaWo, F, D);
		}
		else
		{
			*wi = UniformSampleHemisphere(u);
			*pdf = 1;
			return Vector3f(InvPi);
		}
	}

	__host__ __device__ Vector3f MicrofacetBTDF(const Vector3f &wo, const Vector3f &wi, const Vector3f& F, float D, float G, float upperIOR, float surfaceIOR, bool isRadianceMode)
	{
		if (SameHemisphere(wo, wi)) 
			return Vector3f::Zero();  // transmission only

		float cosThetaO = CosThetaInLocal(wo);
		float cosThetaI = CosThetaInLocal(wi);
		if (cosThetaI == 0 || cosThetaO == 0) return 0.f;

		// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
		float eta = CosThetaInLocal(wo) > 0 ? (surfaceIOR / upperIOR) : (upperIOR / surfaceIOR);
		Vector3f wh = (wo + wi * eta).normalized();
		if (wh.z < 0) wh = -wh;

		float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
		float factor = (isRadianceMode) ? (1 / eta) : 1;

		return (1.f - F) * /*T **/
			std::abs(D * G * eta * eta *
				Abs(Dot(wi, wh)) * Abs(Dot(wo, wh)) * factor * factor /
				(cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
	}

	__host__ __device__ Vector3f SampleMicrofacetBTDF(const Vector3f &wo, Vector3f *wi, const Vector3f &u, float *pdf, float metallic, const Vector3f& R0, float alpha, float upperIOR, float surfaceIOR, bool isRadianceMode)
	{
		if (wo.z == 0)
		{
			*pdf = 0.f;
			return Vector3f::Zero();
		}
			
		Vector3f wh = TrowbridgeReitzSampleWh(wo, u, alpha, alpha);
		float eta = CosThetaInLocal(wo) > 0 ? (upperIOR / surfaceIOR) : (surfaceIOR / upperIOR);

		if (!Refract(wo, wh.normalized(), eta, *wi))
		{
			*pdf = 0.f;
			return Vector3f::Zero();
		}

		float	lambdaWo = TrowbridgeReitzLambda(wo, alpha, alpha),
			lambdaWi = TrowbridgeReitzLambda(*wi, alpha, alpha),
			D = TrowbridgeReitzDistribution(wh, alpha, alpha),
			G = MicrofacetG(lambdaWo, lambdaWi);

		Vector3f F = DisneyFresnelWithSurfaceIOR(metallic, R0, upperIOR, surfaceIOR, Dot(*wi, wh));

		if (u.z > Luminance(F))
		{
			*pdf = MicrofacetPdf(wo, *wi, D, MicrofacetG1(lambdaWo));
			return MicrofacetBTDF(wo, *wi, F, D, G, upperIOR, surfaceIOR, isRadianceMode);
		}
		else
		{
			*wi = UniformSampleHemisphere(u);
			*pdf = 1;
			return Vector3f(InvPi);
		}
	}

	__host__ __device__ float PDFOfMircrofacetBTDF(const Vector3f &wo, const Vector3f &wi, float D, float G1, float upperIOR, float surfaceIOR)
	{
		if (SameHemisphere(wo, wi)) return 0;
		// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
		float eta = CosThetaInLocal(wo) > 0 ? (surfaceIOR / upperIOR) : (upperIOR / surfaceIOR);
		Vector3f wh = (wo + wi * eta).normalized();

		// Compute change of variables _dwh\_dwi_ for microfacet transmission
		float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
		float dwh_dwi = Abs((eta * eta * Dot(wi, wh)) / (sqrtDenom * sqrtDenom));
		return MicrofacetPdf(wo, wh, D, G1) * dwh_dwi;
	}
}
