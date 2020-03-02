#include "Interfaces.h"

#pragma once

namespace RadGrabber
{
	struct Target
	{

	};

	class PathIntegrator : public IIntegrator
	{
	public:

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void Render(const IAggregate & scene) override;
	};

	class MLTIntergrator : public IIntegrator
	{
	public:

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void Render(const IAggregate & scene) override;
	};
}
