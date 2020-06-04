#include "Marshal.cuh"

namespace RadGrabber
{
	namespace Test
	{

		/*
			implicate that hierarchical structure retain more size
		*/
		int MarshalTest()
		{
			printf("FrameInput::%d, FrameInputInternal::%d\n", sizeof(FrameInput), sizeof(FrameInputInternal));
			printf("FrameRequest::%d, FrameRequestMarshal::%d\n", sizeof(FrameRequest), sizeof(FrameRequestMarshal));

			printf("SubmeshDescriptor::%d\n", sizeof(UnitySubMeshDescriptor));

			return 0;
		}
	}
}