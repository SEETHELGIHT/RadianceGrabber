#include <chrono>
#include <ratio>
#include <vector>
#include <cuda.h>
#include <device_functions.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#include "Integrator.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"
#include "Util.h"

namespace RadGrabber
{
	/*
		TODO:: MLT ����
	*/
	/*
		4. russian roulette ���� �߰��� ���߱� ó��?
			���� ���Ŀ�
		5. Subsurface Scattering, Transmission ������ ���߿� ����
	*/

}
