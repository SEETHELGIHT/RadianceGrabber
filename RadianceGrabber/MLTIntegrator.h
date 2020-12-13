#include "DataTypes.cuh"
#include "Integrator.h"

namespace RadGrabber
{
#pragma once

	class MLTIntergrator : public IIntegrator, public ICancelable
	{
	public:
		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void RenderStraight(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;
		__host__ virtual void RenderIncremental(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;

		// ICancelable을(를) 통해 상속됨
		__host__ virtual void ReserveCancel() override;
		__host__ virtual bool IsCancel() override;

	private:
		bool reserveCancel;

	};

	MLTIntergrator* CreateMLTIntegrator(const RequestOption& opt);
}
