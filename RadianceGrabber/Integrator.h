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

		// IIntegrator��(��) ���� ��ӵ�
		__host__ virtual void Render(const IAggregate & scene) override;
	};

	class MLTIntergrator : public IIntegrator
	{
	public:

		// IIntegrator��(��) ���� ��ӵ�
		__host__ virtual void Render(const IAggregate & scene) override;
	};
}
