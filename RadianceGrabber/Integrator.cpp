#include "Integrator.h"
#include "Marshal.h"
#include "DataTypes.cuh"

#pragma pack(push, 4)
struct PathInit
{
	int maxDepth;
};

struct PathSegment2
{
	union
	{
		struct
		{
			int foundLightSourceFlag : 1;
			int endOfDepthFlag : 1;
			int missedRay : 1;
		};
		int endOfRay : 3;
	};
	int remainingBounces : 13;
	int pixelIndex;
	ColorRGBA attenuation;
};

#pragma pack(pop)

__constant__ PathInit pathInitParam;

__host__ void RadGrabber::PathIntegrator::Render(const IAggregate & scene)
{
	
}

__host__ void RadGrabber::MLTIntergrator::Render(const IAggregate & scene)
{
}
