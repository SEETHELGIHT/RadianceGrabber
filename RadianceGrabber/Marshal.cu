#include <cuda_runtime.h>
#include <texture_fetch_functions.hpp>    
#include <texture_fetch_functions.h>    

#include "Marshal.cuh"
#include "DataTypes.cuh"
#include "Sample.cuh"

namespace RadGrabber
{
	__host__ __device__ void CameraChunk::GetRay(const Vector2f& uv, Ray& r)
	{
		Vector3f	nearPos = Vector3f(uv.x, uv.y, 0),
					farPos = Vector3f(uv.x, uv.y, 1);

		nearPos = projectionInverseMatrix.TransformPoint(nearPos);
		nearPos = cameraInverseMatrix.TransformPoint(nearPos);

		farPos = projectionInverseMatrix.TransformPoint(farPos);
		farPos = cameraInverseMatrix.TransformPoint(farPos);

		r.origin = nearPos;
		r.direction = (farPos - nearPos).normalized();
	}

	__host__ __device__ void CameraChunk::GetPixelRay(int pixelIndex, Vector2i s, Ray& r)
	{
		GetRay(Vector2f((float)(pixelIndex % s.x) / s.x, (float)(pixelIndex / s.x) / s.y) * 2 - Vector2f::One(), r);
	}

	__host__ __device__ bool LightChunk::IntersectRay(const Ray & ray, SurfaceIntersection& isect, float& intersectDistance) const
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
			Vector3f	max, pos;

			pos = Vector3f(+width / 2, +height / 2, 0.f);
			transformMatrix.TransformPoint(pos);
			pos += position;
			max = Max(max, pos);

			pos = Vector3f(-width / 2, +height / 2, 0.f);
			transformMatrix.TransformPoint(pos);
			pos += position;
			max = Max(max, pos);

			pos = Vector3f(-width / 2, -height / 2, 0.f);
			transformMatrix.TransformPoint(pos);
			pos += position;
			max = Max(max, pos);

			pos = Vector3f(+width / 2, -height / 2, 0.f);
			transformMatrix.TransformPoint(pos);
			pos += position;
			max = Max(max, pos);

			bb.center = position;
			bb.extents = max - bb.center;
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

	__host__ __device__ bool LightChunk::Sample(const Ray& ray, const SurfaceIntersection& isect, Vector3f& color) const
	{
		switch (type)
		{
		case eUnityLightType::Area:
		case eUnityLightType::Disc:
			color = this->color;
			return true;
		case eUnityLightType::Directional:
		case eUnityLightType::Point:
		case eUnityLightType::Spot:
		default:
			return false;
		}
	}

	// TODO:: microfacet setting
	__device__ void URPLitMaterialChunk::GetMaterialInteract(
		IN const Texture2DChunk* textureBuffer, IN SurfaceIntersection* isect, IN curandState* state,
		INOUT Vector4f& attenuPerComp, INOUT Ray& ray) const
	{
		Vector2f randomSample(curand_uniform(state), curand_uniform(state));

		if (bumpMapIndex >= 0)
		{
			Vector3f bitanWS = Cross(isect->normal, isect->tangent);
			ColorRGBA normTS = textureBuffer[bumpMapIndex].Sample8888(isect->uv);
			isect->normal = normTS.r * bitanWS + normTS.g * isect->normal + normTS.b * isect->tangent;
		}
			
		//if (IsMetallicSetup())
		{
			float smoothness = SampleSmoothness(textureBuffer, isect->uv);
			float metallic = SampleMetallic(textureBuffer, isect->uv);
			
			//if (IsOpaque())
			{
				if (IsAlphaClipping() && randomSample.x >= textureBuffer[baseMapIndex].Sample8888(isect->uv).r)
				{
					goto METALLIC_TRANSPARENT_INTERACT;
				}

				ray.origin = isect->position;

				// metal
				if (randomSample.x < metallic)
				{
					ray.direction = Reflect(ray.direction, isect->normal);

					Vector3f randDirection = UniformSampleSphere(randomSample);
					if (randDirection != ray.direction)
						ray.direction = (ray.direction + (1.f - smoothness) * randDirection).normalized();
				}
				// dielectric
				else
				{
					ray.direction = UniformSampleHemisphereInFrame(isect->normal, randomSample);
				}
				
				Vector3f albedo = SampleAlbedo(textureBuffer, isect->uv);
				attenuPerComp *= Vector4f(albedo.x, albedo.y, albedo.z, 1.f);
			}
			// transparent 
			//else
			{
METALLIC_TRANSPARENT_INTERACT:
			}
		}
		//else
		{
			/*
				specular
			*/
		}
	}

	__device__ void MaterialChunk::GetMaterialInteract(IN const Texture2DChunk* textureBuffer, IN SurfaceIntersection* isect, IN curandState* state, INOUT Vector4f& attenuPerComp, INOUT Ray& ray) const
	{
		switch (type)
		{
		case eShaderType::UniversalLit:
			URPLit.GetMaterialInteract(textureBuffer, isect, state, attenuPerComp, ray);
			break;
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
			int	fx = floor(uv.x * size.x), cx = ceil(uv.x * size.x),
				fy = floor(uv.y * size.y), cy = ceil(uv.y * size.y);
			ColorRGBA	fyc = ((ColorRGBA)colorPtr[fy * size.x + fx] + (ColorRGBA)colorPtr[fy * size.x + cx]) / 2,
				cyc = ((ColorRGBA)colorPtr[cy * size.x + fx] + (ColorRGBA)colorPtr[cy * size.x + cx]) / 2;
			return (fyc + cyc) / 2;
		}
		case eUnityFilterMode::Trilinear:
		default:
			ASSERT(false);
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
		}
#else // if __CUDA_ARCH__
		float4 tex;
		tex2D(&tex, (cudaTextureObject_t)pixelPtr, uv.x, uv.y);
		return ColorRGBA(tex.x, tex.y, tex.z, tex.w);
#endif
	}

}
