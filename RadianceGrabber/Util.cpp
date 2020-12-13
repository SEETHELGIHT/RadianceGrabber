#include "Util.h"
#include "DataTypes.cuh"

#include <ctime>
#include <cstdarg>
#include <mutex>

namespace RadGrabber
{
	FILE* g_LogPtr = nullptr;
	int g_BlockLog = 0;
	std::mutex g_LogMutex;

	void Log(const char* format, ...)
	{
		std::lock_guard<std::mutex> lock(g_LogMutex);

		if (g_BlockLog) return;
		if (g_LogPtr == nullptr)
		{
			time_t rawtime;
			struct tm timeinfo;
			char buffer[80];

			time(&rawtime);
			localtime_s(&timeinfo, &rawtime);

			strftime(buffer, sizeof(buffer), "./log_%d_%m_%Y-%H_%M_%S.txt", &timeinfo);

			fopen_s(&g_LogPtr, buffer, "at");
		}

		assert(g_LogPtr != nullptr);

		va_list arg_ptr;
		va_start(arg_ptr, format);
		vfprintf(g_LogPtr, format, arg_ptr);
		va_end(arg_ptr);

		fflush(g_LogPtr);
	}

	void SetFilePtr(FILE* ptr) { g_LogPtr = ptr; }
	FILE* GetFilePtr() { return g_LogPtr; }

	void __declspec(dllexport) __stdcall SetBlockLog(int isBlock)
	{
		g_BlockLog = isBlock;
	}
	void __declspec(dllexport) __stdcall FlushLog()
	{
		if (g_LogPtr != nullptr)
		{
			std::lock_guard<std::mutex> lock(g_LogMutex);
			fclose(g_LogPtr);
			g_LogPtr = nullptr;
		}
	}
}
