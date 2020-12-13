#pragma comment(lib, "RadianceGrabber.lib")

#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cassert>
#include <cstring>
#include <filesystem>

#include "ConfigUtility.h"
#include "Util.h"

namespace RadGrabber
{
	namespace Test
	{
		//int SphereTest();
		int LogFunctionTest();
		int MarshalTest();
		int RayIntersect();
		int MeshTest();
		int MatrixTest();
		int NormalMappingTest();
		int RayFromCamera();
		int SaveFrameRequestTest();
		int FrameDataValidateTest();
		//int HostDeviceCopyTest();
		int UniformRandomSampleTest();
		int VectorFormulaTest();
		int StaticMeshBuildTest();
		int TransformBVHBuildTest();
		int IterativeTransformBVHBuildTest();
		int IterativeStaticMeshTraversalTest();
		int AccelAggregateIntersectTest();
		int HitTestKernelTest();
		int PixelToPNGTest();
		int PNGToMJPGTest();
		int SingleRayTest();
		int MicrofacetFunctionTest();

		int PathIntegratorAndLinearAggTest();
		int PathIntegratorAndAccelAggTest();
		int PathIntegratorIncrementalAndStraight();
		int PathIntegratorStraigntAndIncremental();
		int IterativePathIntegratorAndAccelAggTest();
		int PathIntegratorMultiFrameRequestProc();
		int SIngleFrameGenThreadManagementTest();
		int MultiFrameGenThreadManagementTest();

		struct TestChunk {
			char testName[257];
			int (*testFunc)();

			TestChunk(const char* testName, int(*testFunc)()) : testFunc(testFunc)
			{
				strcpy(this->testName, testName);
			}
		};
		TestChunk* g_ModuleTest[] = {
			// 0
			new TestChunk("PathIntegratorAndLinearAggTest", PathIntegratorAndLinearAggTest),
			new TestChunk("PathIntegratorAndAccelAggTest", PathIntegratorAndAccelAggTest),
			new TestChunk("SIngleFrameGenThreadManagementTest ", SIngleFrameGenThreadManagementTest),
			new TestChunk("PathIntegratorIncreVSStraight0", PathIntegratorIncrementalAndStraight),
			new TestChunk("PathIntegratorIncreVSStraight1", PathIntegratorStraigntAndIncremental),
			// 5
			new TestChunk("IterativePathIntegratorAndAccelAggTest", IterativePathIntegratorAndAccelAggTest),
			new TestChunk("PathIntegratorMultiFrameRequestProc", PathIntegratorMultiFrameRequestProc),
			new TestChunk("MultiFrameGenThreadManagementTest ", MultiFrameGenThreadManagementTest),
		};
		TestChunk* g_CalcTest[] = {
			// 0
			nullptr,
			//new TestChunk("SphereTest", SphereTest),
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
			//new TestChunk("HostDeviceCopyTest", HostDeviceCopyTest),
			new TestChunk("StaticMeshBuildTest", StaticMeshBuildTest),
			new TestChunk("TransformBVHBuildTest", TransformBVHBuildTest),
			new TestChunk("UniformRandomSampleTest", UniformRandomSampleTest),
			new TestChunk("VectorFormulaTest", VectorFormulaTest),
			new TestChunk("IterativeTransformBVHBuildTest", IterativeTransformBVHBuildTest),
			// 15
			new TestChunk("IterativeStaticMeshTraversalTest", IterativeStaticMeshTraversalTest),
			new TestChunk("AccelAggregateIntersectTest", AccelAggregateIntersectTest),
			new TestChunk("HitTestKernelTest", HitTestKernelTest),
			new TestChunk("PixelToPNGTest", PixelToPNGTest),
			new TestChunk("PNGToMJPGTest", PNGToMJPGTest),
			// 20
			new TestChunk("SingleRayTest", SingleRayTest),
			new TestChunk("MicrofacetFunctionTest", MicrofacetFunctionTest),
		};
	}
}

int main(int argc, char** argv)
{
	{
		time_t rawtime;
		struct tm timeinfo;

		time(&rawtime);
		localtime_s(&timeinfo, &rawtime);

		char buffer[1025];
		strftime(buffer, sizeof(buffer), "%Y:%m:%d-%H:%M:%S", &timeinfo);
		puts(buffer);
	}

	RadGrabber::Utility::TestProjectConfig config;
	config.RefreshValues();

	int mode = config.testMode;
	bool moduleTest = mode < 100;
	RadGrabber::Test::TestChunk* t;
	if (!moduleTest)
	{
		mode -= 100;
		t = RadGrabber::Test::g_CalcTest[mode];
	}
	else
		t = RadGrabber::Test::g_ModuleTest[mode];
	
	{
		RadGrabber::TimeProfiler p(t->testName);
		if (RadGrabber::GetFilePtr())
			p.outFile = RadGrabber::GetFilePtr();
		puts(t->testName);
		t->testFunc();
		p.Print();
	}

	getchar();

	return 0;
}
