#include "DataTypes.cuh"

#pragma once

namespace RadGrabber
{

#pragma region Inline Trigonometic Functions For BSDF(x=tangent, y=bitangent, z=normal)

	__forceinline__ __host__ __device__ float CosThetaInLocal(const Vector3f &w) { return w.z; }
	__forceinline__ __host__ __device__ float Cos2ThetaInLocal(const Vector3f &w) { return w.z * w.z; }
	__forceinline__ __host__ __device__ float AbsCosThetaInLocal(const Vector3f &w) { return Abs(w.z); }
	__forceinline__ __host__ __device__ float Sin2ThetaInLocal(const Vector3f &w) {
		return fmax((float)0, (float)1 - Cos2ThetaInLocal(w));
	}
	__forceinline__ __host__ __device__ float SinThetaInLocal(const Vector3f &w) { return sqrtf(Sin2ThetaInLocal(w)); }
	__forceinline__ __host__ __device__ float TanThetaInLocal(const Vector3f &w) { return SinThetaInLocal(w) / CosThetaInLocal(w); }
	__forceinline__ __host__ __device__ float Tan2ThetaInLocal(const Vector3f &w) {
		return Sin2ThetaInLocal(w) / Cos2ThetaInLocal(w);
	}

	__forceinline__ __host__ __device__ float CosPhiInLocal(const Vector3f &w) {
		float sinTheta = SinThetaInLocal(w);
		return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1.f, 1.f);
	}
	__forceinline__ __host__ __device__ float SinPhiInLocal(const Vector3f &w) {
		float sinTheta = SinThetaInLocal(w);
		return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1.f, 1.f);
	}

	__forceinline__ __host__ __device__ float Cos2PhiInLocal(const Vector3f &w) { return CosPhiInLocal(w) * CosPhiInLocal(w); }
	__forceinline__ __host__ __device__ float Sin2PhiInLocal(const Vector3f &w) { return SinPhiInLocal(w) * SinPhiInLocal(w); }
	__forceinline__ __host__ __device__ float CosDPhiInLocal(const Vector3f &wa, const Vector3f &wb) {
		return Clamp(
			(wa.x * wb.x + wa.y * wb.y) / sqrtf((wa.x * wa.x + wa.y * wa.y) *
			(wb.x * wb.x + wb.y * wb.y)),
			-1.f, 1.f);
	}

	__forceinline__ __host__ __device__ bool SameHemisphere(const Vector3f &w, const Vector3f &wp) 
	{
		return w.z * wp.z > 0;
	}

#pragma endregion Inline Trigonometic Functions For BSDF

#pragma region Microfacet Functions

	// BxDF Utility Functions
	__forceinline__ __host__ __device__ float FrDielectric(float cosThetaI, float etaI, float etaT)
	{
		cosThetaI = Clamp(cosThetaI, -1.f, 1.f);
		// Potentially swap indices of refraction
		bool entering = cosThetaI > 0.f;
		if (!entering) {
			float tempETA = etaI;
			etaI = etaT;
			etaT = tempETA;
			cosThetaI = Abs(cosThetaI);
		}

		// Compute _cosThetaT_ using Snell's law
		float sinThetaI = sqrtf(fmax((float)0, 1 - cosThetaI * cosThetaI));
		float sinThetaT = etaI / etaT * sinThetaI;

		// Handle total internal reflection
		if (sinThetaT >= 1) return 1;
		float cosThetaT = sqrtf(fmax((float)0, 1 - sinThetaT * sinThetaT));
		float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
			((etaT * cosThetaI) + (etaI * cosThetaT));
		float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
			((etaI * cosThetaI) + (etaT * cosThetaT));
		return (Rparl * Rparl + Rperp * Rperp) / 2;
	}

	// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
	__forceinline__ __host__ __device__ Vector3f FrConductor(float cosThetaI, const Vector3f &etai, const Vector3f &etat, const Vector3f &k)
	{
		cosThetaI = Clamp(cosThetaI, -1.f, 1.f);
		Vector3f eta = etat / etai;
		Vector3f etak = k / etai;

		float cosThetaI2 = cosThetaI * cosThetaI;
		float sinThetaI2 = 1.f - cosThetaI2;
		Vector3f eta2 = eta * eta;
		Vector3f etak2 = etak * etak;

		Vector3f t0 = eta2 - etak2 - sinThetaI2;
		Vector3f a2plusb2 = Sqrtv(t0 * t0 + 4.f * eta2 * etak2);
		Vector3f t1 = a2plusb2 + cosThetaI2;
		Vector3f a = Sqrtv(0.5f * (a2plusb2 + t0));
		Vector3f t2 = (float)2 * cosThetaI * a;
		Vector3f Rs = (t1 - t2) / (t1 + t2);

		Vector3f t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
		Vector3f t4 = t2 * sinThetaI2;
		Vector3f Rp = Rs * (t3 - t4) / (t3 + t4);

		return 0.5f * (Rp + Rs);
	}

	__forceinline__ __host__ __device__ float MicrofacetG1(float lambdaW) { return 1 / (1 + lambdaW); }
	__forceinline__ __host__ __device__ float MicrofacetG(float lambdaWo, float lambdaWi) { return 1 / (1 + lambdaWo + lambdaWi); }
	__forceinline__ __host__ __device__ float DisneyMicrofacetG(float lambdaWo, float lambdaWi) { return 1 / (1 + lambdaWo) / (1 + lambdaWi); }
	__forceinline__ __host__ __device__ float MicrofacetPdf(const Vector3f &wo, const Vector3f &wh, float D, float G1)
	{
		return D * G1 * Abs(Dot(wo, wh)) / AbsCosThetaInLocal(wo);
	}

	__forceinline__ __host__ __device__ void TrowbridgeReitzSample11(float cosTheta, float U1, float U2, float *slope_x, float *slope_y)
	{
		// special case (normal incidence)
		if (cosTheta > .9999) {
			float r = sqrt(U1 / (1 - U1));
			float phi = 6.28318530718 * U2;
			*slope_x = r * cos(phi);
			*slope_y = r * sin(phi);
			return;
		}

		float sinTheta = sqrtf(fmax((float)0, (float)1 - cosTheta * cosTheta));
		float tanTheta = sinTheta / cosTheta;
		float a = 1 / tanTheta;
		float G1 = 2 / (1 + sqrtf(1.f + 1.f / (a * a)));

		// sample slope_x
		float A = 2 * U1 / G1 - 1;
		float tmp = 1.f / (A * A - 1.f);
		if (tmp > 1e10) tmp = 1e10;
		float B = tanTheta;
		float D = sqrtf(fmax(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
		float slope_x_1 = B * tmp - D;
		float slope_x_2 = B * tmp + D;
		*slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

		// sample slope_y
		float S;
		if (U2 > 0.5f) {
			S = 1.f;
			U2 = 2.f * (U2 - .5f);
		}
		else {
			S = -1.f;
			U2 = 2.f * (.5f - U2);
		}
		float z =
			(U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
			(U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
		*slope_y = S * z * sqrtf(1.f + *slope_x * *slope_x);

		ASSERT(!isinf(*slope_y));
		ASSERT(!isnan(*slope_y));
	}

	__forceinline__ __host__ __device__ Vector3f TrowbridgeReitzSample(const Vector3f &wi, float alpha_x, float alpha_y, float U1, float U2)
	{
		// 1. stretch wi
		Vector3f wiStretched = Vector3f(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalized();

		// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
		float slope_x, slope_y;
		TrowbridgeReitzSample11(CosThetaInLocal(wiStretched), U1, U2, &slope_x, &slope_y);

		// 3. rotate
		float tmp = CosPhiInLocal(wiStretched) * slope_x - SinPhiInLocal(wiStretched) * slope_y;
		slope_y = SinPhiInLocal(wiStretched) * slope_x + CosPhiInLocal(wiStretched) * slope_y;
		slope_x = tmp;

		// 4. unstretch
		slope_x = alpha_x * slope_x;
		slope_y = alpha_y * slope_y;

		// 5. compute normal
		return Vector3f(-slope_x, -slope_y, 1.).normalized();
	}

	__forceinline__ __host__ __device__ float TrowbridgeReitzDistribution(const Vector3f &wh, float alphax, float alphay)
	{
		float tan2Theta = Tan2ThetaInLocal(wh);
		if (isinf(tan2Theta)) return 0.;
		const float cos4Theta = Cos2ThetaInLocal(wh) * Cos2ThetaInLocal(wh);
		float e =
			(Cos2PhiInLocal(wh) / (alphax * alphax) + Sin2PhiInLocal(wh) / (alphay * alphay)) *
			tan2Theta;
		return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
	}

	__forceinline__ __host__ __device__ float TrowbridgeReitzLambda(const Vector3f &w, float alphax, float alphay)
	{
		float absTanTheta = Abs(TanThetaInLocal(w));
		if (isinf(absTanTheta)) return 0.;
		// Compute _alpha_ for direction _w_
		float alpha =
			sqrtf(Cos2PhiInLocal(w) * alphax * alphax + Sin2PhiInLocal(w) * alphay * alphay);
		float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
		return (-1 + sqrtf(1.f + alpha2Tan2Theta)) / 2;
	}

	__forceinline__ __host__ __device__ Vector3f TrowbridgeReitzSampleWh(const Vector3f &wo, const Vector2f &u, float alphax, float alphay)
	{
		Vector3f wh;
		bool flip = wo.z < 0;
		wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
		if (flip) wh = -wh;
		return wh;
	}
	__forceinline__ __host__ __device__ float TrowbridgeReitzRoughnessToAlpha(float roughness) 
	{
		roughness = fmax(roughness, (float)1e-3);
		float x = log(roughness);
		return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
			0.000640711f * x * x * x * x;
	}

	__forceinline__ __host__ __device__ float SchlickWeight(float cosTheta) {
		float m = Clamp(1 - cosTheta, 0.f, 1.f);
		return (m * m) * (m * m) * m;
	}

	__forceinline__ __host__ __device__ float FrSchlick(float R0, float cosTheta) {
		return Lerp(R0, 1, SchlickWeight(cosTheta));
	}

	__forceinline__ __host__ __device__ Vector3f FrSchlick(const Vector3f &R0, float cosTheta) {
		return Lerp(R0, Vector3f(1.f), SchlickWeight(cosTheta));
	}

	__forceinline__ __host__ __device__ Vector3f DisneyFresnel(float metallic, Vector3f R0, float surfaceIOR, float cosTheta)
	{
		return Lerp(Vector3f(FrDielectric(cosTheta, 1, surfaceIOR)), FrSchlick(R0, cosTheta), metallic);
	}

	__forceinline__ __host__ __device__ Vector3f DisneyFresnelWithSurfaceIOR(float metallic, Vector3f R0, float mediumIOR, float surfaceIOR, float cosTheta)
	{
		return Lerp(Vector3f(FrDielectric(cosTheta, mediumIOR, surfaceIOR)), FrSchlick(R0, cosTheta), metallic);
	}
	
	__host__ __device__ Vector3f MicrofacetBRDF(const Vector3f &wo, const Vector3f &wi, float alpha, float lambdaWo, const Vector3f& F, float D);
	__host__ __device__ float PDFOfMicrofacetBRDF(const Vector3f &wo, const Vector3f &wi, float D, float G1);
	__host__ __device__ Vector3f SampleMicrofacetBRDF(const Vector3f &wo, Vector3f *wi, const Vector3f &u, float *pdf, float metallic, const Vector3f& R0, float alpha, float upperIOR, float surfaceIOR);

	__host__ __device__ Vector3f MicrofacetBTDF(const Vector3f &wo, const Vector3f &wi, const Vector3f& F, float D, float G, float upperIOR, float surfaceIOR, bool isRadianceMode);
	__host__ __device__ Vector3f SampleMicrofacetBTDF(const Vector3f &wo, Vector3f *wi, const Vector3f &u, float *pdf, float metallic, const Vector3f& R0, float alpha, float upperIOR, float surfaceIOR, bool isRadianceMode);	
	__host__ __device__ float PDFOfMircrofacetBTDF(const Vector3f &wo, const Vector3f &wi, float D, float G1, float upperIOR, float surfaceIOR);

#pragma endregion Microfacet Functions
}
