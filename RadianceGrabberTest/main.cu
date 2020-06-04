#pragma comment(lib, "RadianceGrabber.lib")

#include <cstdio>
#include <cassert>
#include <cstring>
#include "ConfigUtility.h"

namespace RadGrabber
{
	namespace Test
	{
		int SphereTest();
		int LogFunctionTest();
		int MarshalTest();
		int RayIntersect();
		int MeshTest();
		int MatrixTest();
		int NormalMappingTest();
		int RayFromCamera();
		int SaveFrameRequestTest();
		int FrameDataValidateTest();
		int HostDeviceCopyTest();
		int StaticMeshBuildTest();
		int TransformBVHBuildTest();
		int UniformRandomSampleTest();
		int VectorFormulaTest();

		int PathIntegratorAndLinearAggTest();
		int PathIntegratorAndAccelAggTest();
		int ThreadManagementTest();

		struct TestChunk {
			char testName[257];
			int (*testFunc)();

			TestChunk(const char* testName, int(*testFunc)()) : testFunc(testFunc)
			{
				strcpy(this->testName, testName);
			}
		};
		TestChunk* g_ModuleTest[] = {
			new TestChunk("PathIntegratorAndLinearAggTest", PathIntegratorAndLinearAggTest),
			new TestChunk("PathIntegratorAndAccelAggTest", PathIntegratorAndAccelAggTest),
			new TestChunk("ThreadManagementTest ", ThreadManagementTest),
		};
		TestChunk* g_CalcTest[] = {
			// 0
			new TestChunk("SphereTest", SphereTest),
			new TestChunk("LogFunctionTest", LogFunctionTest),
			new TestChunk("MarshalTest", MarshalTest),
			new TestChunk("RayIntersect", RayIntersect),
			new TestChunk("MeshTest", MeshTest),
			// 5
			new TestChunk("MatrixTest", MatrixTest),
			new TestChunk("NormalMappingTest", NormalMappingTest),
			new TestChunk("RayFromCamera", RayFromCamera),
			new TestChunk("SaveFrameRequestTest", SaveFrameRequestTest),
			new TestChunk("FrameDataValidateTest", FrameDataValidateTest),
			// 10
			new TestChunk("HostDeviceCopyTest", HostDeviceCopyTest),
			new TestChunk("StaticMeshBuildTest", StaticMeshBuildTest),
			new TestChunk("TransformBVHBuildTest", TransformBVHBuildTest),
			new TestChunk("UniformRandomSampleTest", UniformRandomSampleTest),
			new TestChunk("VectorFormulaTest", VectorFormulaTest),
		};
	}
}

int main()
{
	RadGrabber::Utility::ConfigUtility::RefreshValues();
	
	int mode = RadGrabber::Utility::ConfigUtility::GetMode();
	
	if (mode < 100)
	{
		puts(RadGrabber::Test::g_ModuleTest[mode]->testName);
		RadGrabber::Test::g_ModuleTest[mode]->testFunc();
	}
	else
	{
		mode -= 100;
		puts(RadGrabber::Test::g_ModuleTest[mode]->testName);
		RadGrabber::Test::g_CalcTest[mode]->testFunc();
	}
}
