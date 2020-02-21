#include "Marshal.h"

using namespace RadGrabber;

__host__ __device__ bool LightChunk::IntersectRay(const Ray & ray, SurfaceIntersection& isect, float& intersectDistance)
{
	/*	
		TODO:: light color
	*/

	switch (type)
	{
	case eUnityLightType::Area:
		return true;
	case eUnityLightType::Disc:
		return true;
	case eUnityLightType::Directional:
	case eUnityLightType::Point:
	case eUnityLightType::Spot:
	default:
		return false;
	}

}

__host__ __device__ bool LightChunk::GetBoundingBox(Bounds& bb)
{
	/*
		TODO:: calc aabb
	*/

	switch (type)
	{
	case eUnityLightType::Area:
		return true;
	case eUnityLightType::Disc:
		return true;
	case eUnityLightType::Directional:
	case eUnityLightType::Point:
	case eUnityLightType::Spot:
	default:
		return false;
	}
}