#include <cuda_runtime.h>
#include <texture_fetch_functions.hpp>    
#include <texture_fetch_functions.h>    

#include "Marshal.cuh"
#include "DataTypes.cuh"
#include "Sample.cuh"

namespace RadGrabber
{
	__host__ __device__ bool LightChunk::IntersectRay(const Ray & rayInWS, SurfaceIntersection& isect, float& intersectDistance) const
	{
		Ray rayInMS = Ray(
			this->transformInverseMatrix.TransformPoint(rayInWS.origin),
			this->transformInverseMatrix.TransformVector(rayInWS.direction)
		);
		/*
			ray-plane intersection
			R(t) = r0 + td, (p - p0)*n = 0, p is intersection point on plane, p0 is other point on plane
			1. (r0 + td - p0)*n = 0
			2. td*n + (r0-p0)*n = 0
			3. td*n = (p0-r0)*n
			4. t = (p0-r0)*n / d*n
		*/
		switch (type)
		{
		case eUnityLightType::Area:
		{
			float denom = Dot(-rayInWS.direction, forward);
			if (denom <= 0)
				return false;

			float t = Dot((rayInWS.origin - position), forward) / denom;
			if (t <= 0)
				return false;

			Vector3f delta = rayInWS.origin + rayInWS.direction * t - position;
			quaternion.Inversed().Rotate(delta);
			delta = Abs(delta);

			if (delta.x <= width && delta.y <= height)
			{
				isect.isHit = 1;
				isect.isGeometry = 0;
				intersectDistance = t;
				return true;
			}
			else
				return false;
		}
		case eUnityLightType::Disc:
		{
			float denom = Dot(-rayInWS.direction, forward);
			if (denom <= 0)
				return false;

			float t = Dot((rayInWS.origin - position), forward) / denom;
			if (t <= 0)
				return false;

			float sqrMag = ((rayInWS.origin + rayInWS.direction * t) - position).sqrMagnitude();
			if (sqrMag <= range * range)
			{
				isect.isHit = 1;
				isect.isGeometry = 0;
				intersectDistance = t;
				return true;
			}
			else
				return false;
		}
		case eUnityLightType::Directional:
		case eUnityLightType::Point:
		case eUnityLightType::Spot:
		default:
			isect.isHit = 0;
			return false;
		}
	}

	__host__ __device__ bool LightChunk::IntersectRay(const Ray & rayInWS, float minDistance, float maxDistance, float& intersectDistance) const
	{
		Ray rayInMS = Ray(
			this->transformInverseMatrix.TransformPoint(rayInWS.origin),
			this->transformInverseMatrix.TransformVector(rayInWS.direction)
		);
		/*
			ray-plane intersection
			R(t) = r0 + td, (p - p0)*n = 0, p is intersection point on plane, p0 is other point on plane
			1. (r0 + td - p0)*n = 0
			2. td*n + (r0-p0)*n = 0
			3. td*n = (p0-r0)*n
			4. t = (p0-r0)*n / d*n
		*/
		switch (type)
		{
		case eUnityLightType::Area:
		{
			float denom = Dot(-rayInWS.direction, forward);
			if (denom <= 0)
				return false;

			float t = Dot((rayInWS.origin - position), forward) / denom;
			if (t <= 0)
				return false;
			if (minDistance > intersectDistance || intersectDistance >= maxDistance)
				return false;

			Vector3f delta = rayInWS.origin + rayInWS.direction * t - position;
			quaternion.Inversed().Rotate(delta);
			delta = Abs(delta);

			if (delta.x <= width && delta.y <= height)
			{
				intersectDistance = t;
				return true;
			}
			else
				return false;
		}
		case eUnityLightType::Disc:
		{
			float denom = Dot(-rayInWS.direction, forward);
			if (denom <= 0)
				return false;

			float t = Dot((rayInWS.origin - position), forward) / denom;
			if (t <= 0)
				return false;
			if (minDistance > intersectDistance || intersectDistance >= maxDistance)
				return false;

			float sqrMag = ((rayInWS.origin + rayInWS.direction * t) - position).sqrMagnitude();
			if (sqrMag <= range * range)
			{
				intersectDistance = t;
				return true;
			}
			else
				return false;
		}
		case eUnityLightType::Directional:
		case eUnityLightType::Point:
		case eUnityLightType::Spot:
		default:
			return false;
		}
	}

	__host__ __device__ bool LightChunk::IntersectRayOnly(const Ray & ray) const
	{
		/*
			ray-plane intersection
			1. R(t) = r0 + td, (p - p0)*n = 0, p is intersection point on plane, p0 is other point on plane
			2. (r0 + td - p0)*n = 0
			3. td*n + (r0-p0)*n = 0
			4. td*n = (p0-r0)*n
			5. t = (p0-r0)*n / d*n
		*/
		switch (type)
		{
		case eUnityLightType::Area:
		{
			float denom = Dot(-ray.direction, forward);
			if (denom <= 0)
				return false;

			float t = Dot((position - ray.origin), forward) / denom;
			if (t <= 0)
				return false;

			Vector3f delta = ray.origin + ray.direction * t - position;
			quaternion.Inversed().Rotate(delta);

			return delta.x <= width && delta.y <= height;
		}
		case eUnityLightType::Disc:
		{
			float denom = Dot(-ray.direction, forward);
			if (denom <= 0)
				return false;

			float t = Dot((position - ray.origin), forward) / denom;
			if (t <= 0)
				return false;

			float sqrMag = ((ray.origin + ray.direction * t) - position).sqrMagnitude();
			return sqrMag <= range * range;
		}
		case eUnityLightType::Directional:
		case eUnityLightType::Point:
		case eUnityLightType::Spot:
		default:
			return false;
		}
	}

	__host__ __device__ bool LightChunk::GetBoundingBox(Bounds& bb) const
	{
		switch (type)
		{
		case eUnityLightType::Area:
		{
			Vector3f	maxVal = Vector3f::Zero(), pos;

			pos = transformMatrix.TransformPoint(Vector3f(+width / 2, +height / 2, 0.f));
			maxVal = Max(maxVal, pos);

			pos = transformMatrix.TransformPoint(Vector3f(-width / 2, +height / 2, 0.f));
			maxVal = Max(maxVal, pos);

			pos = transformMatrix.TransformPoint(Vector3f(-width / 2, -height / 2, 0.f));
			maxVal = Max(maxVal, pos);

			pos = transformMatrix.TransformPoint(Vector3f(+width / 2, -height / 2, 0.f));
			maxVal = Max(maxVal, pos);

			bb.center = position;
			bb.extents = maxVal - bb.center + Vector3f(EPSILON, EPSILON, EPSILON);

			return true;
		}			
		case eUnityLightType::Disc:
			bb.center = position;
			bb.extents = this->range * Sqrtv(Vector3f::One() - (this->forward * this->forward));
			return true;
		case eUnityLightType::Directional:
		case eUnityLightType::Point:
		case eUnityLightType::Spot:
			return false;
		default:
			return false;
		}
	}

	__host__ __device__ ColorRGBA Texture2DChunk::Sample8888(const Vector2f& uv) const
	{
#ifndef __CUDA_ARCH__
		ColorRGBA32* colorPtr = (ColorRGBA32*)pixelPtr;

		switch (filter)
		{
		case eUnityFilterMode::Point:
		{
			Vector2i roundedUV = Round(Vector2f(uv.x * size.x, uv.y * size.y));
			return colorPtr[roundedUV.y * size.x + roundedUV.x];
		}
		case eUnityFilterMode::Bilinear:
		{
			int	fx = int(uv.x * size.x), cx = int(uv.x * size.x + 1),
				fy = int(uv.y * size.y), cy = int(uv.y * size.y + 1);
			ColorRGBA	fyc = ((ColorRGBA)colorPtr[fy * size.x + fx] + (ColorRGBA)colorPtr[fy * size.x + cx]) / 2,
				cyc = ((ColorRGBA)colorPtr[cy * size.x + fx] + (ColorRGBA)colorPtr[cy * size.x + cx]) / 2;
			return (fyc + cyc) / 2;
		}
		case eUnityFilterMode::Trilinear:
		default:
			ASSERT(false);
			return ColorRGBA(0, 0, 0, 0);
		}
#else // if __CUDA_ARCH__
		float4 tex = tex2D<float4>((cudaTextureObject_t)pixelPtr, uv.x, uv.y);
		return ColorRGBA(tex.x, tex.y, tex.z, tex.w);
#endif
	}


	__host__ __device__ ColorRGBA Texture2DChunk::Sample32323232(const Vector2f& uv) const
	{
#ifndef __CUDA_ARCH__
		ColorRGBA32* colorPtr = (ColorRGBA32*)pixelPtr;

		switch (filter)
		{
		case eUnityFilterMode::Point:
		{
			Vector2i roundedUV = Round(Vector2f(uv.x * size.x, uv.y * size.y));
			return colorPtr[roundedUV.y * size.x + roundedUV.x];
		}
		case eUnityFilterMode::Bilinear:
		{
			int	fx = floor(uv.x * size.x), cx = ceil(uv.x * size.x),
				fy = floor(uv.y * size.y), cy = ceil(uv.y * size.y);
			ColorRGBA	fyc = ((ColorRGBA)colorPtr[fy * size.x + fx] + (ColorRGBA)colorPtr[fy * size.x + cx]) / 2,
				cyc = ((ColorRGBA)colorPtr[cy * size.x + fx] + (ColorRGBA)colorPtr[cy * size.x + cx]) / 2;
			return (fyc + cyc) / 2;
		}
		case eUnityFilterMode::Trilinear:
		default:
			ASSERT(false);
			return ColorRGBA(0, 0, 0, 0);
		}
#else // if __CUDA_ARCH__
		float4 tex;
		tex2D(&tex, (cudaTextureObject_t)pixelPtr, uv.x, uv.y);
		return ColorRGBA(tex.x, tex.y, tex.z, tex.w);
#endif
	}

}
