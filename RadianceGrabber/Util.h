
#include <memory>
#include <chrono>
#include "Define.h"

#pragma once

namespace RadGrabber
{
#ifndef REMOVE_LOG
	void Log(const char* format, ...);
	void __declspec(dllexport) __stdcall SetBlockLog(int isBlock);
	void SetFilePtr(FILE* ptr);
	FILE* GetFilePtr();
	void __declspec(dllexport) __stdcall FlushLog();
#else
#define Log(x, ...) 
#define SetBlockLog(b) 
#define SetFilePtr(p) 
#define FlushLog() 
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			Log("Assertion: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) assert(code);
		}
	}

	inline void* MAllocDevice(int cb)
	{
		void* ptr = nullptr;
		gpuErrchk(cudaMalloc(&ptr, cb));
		return ptr;
	}

	inline void* MAllocManaged(int cb)
	{
		void* ptr = nullptr;
		gpuErrchk(cudaMallocManaged(&ptr, cb));
		return ptr;
	}

	inline void* MAllocHost(int cb, int flags = 0U)
	{
		return malloc(cb);
		void* ptr = nullptr;
		gpuErrchk(cudaMallocHost(&ptr, cb, flags));
		return ptr;
	}

	template<typename T>
	inline T* MAllocHost(int count, int flags = 0U)
	{
		T* ptr = nullptr;
		gpuErrchk(cudaMallocHost(&ptr, count * sizeof(T), flags));
		return ptr;
	}

	template<typename T>
	inline void MCopy(void* dst, const void* src, int count, cudaMemcpyKind kind)
	{
		gpuErrchk(cudaMemcpy(dst, src, count * sizeof(T), kind));
	}

	inline void MCopy(void* dst, const void* src, int size, cudaMemcpyKind kind)
	{
		gpuErrchk(cudaMemcpy(dst, src, size, kind));
	}

	inline void MCopyToSymbol(void* dst, const void* src, int size)
	{
		gpuErrchk(cudaMemcpyToSymbol(dst, src, size));
	}

	inline void MSet(void* dst, int value, size_t count)
	{
		gpuErrchk(cudaMemset(dst, value, count));
	}

#ifndef RELEASE
	struct TimeProfiler
	{
		bool printed;
		FILE* outFile;
		std::chrono::system_clock::time_point pt;
		char* printString;

		TimeProfiler(const char* string) : outFile(stdout), pt(std::chrono::system_clock::now()), printed(false)
		{
			printString = (char*)malloc(sizeof(char*) * (strlen(string) + 1));
			strcpy_s(printString, (strlen(string) + 1), string);
		}
		TimeProfiler(FILE* fp, const char* string) : outFile(fp), pt(std::chrono::system_clock::now()), printed(false)
		{
			printString = (char*)malloc(sizeof(char*) * (strlen(string) + 1));
			strcpy_s(printString, (strlen(string) + 1), string);
		}
		static void PrintTime(FILE* fp, const long long int& time)
		{
			if (time < 1000)
				fprintf(fp, "%lldnsec", time);
			else if (time < 1000 * 1000)
				fprintf(fp, "%.3lfusec", time / 1000.0);
			else if (time < 1000 * 1000 * 1000)
				fprintf(fp, "%.3lfmsec", time / 1000000.0);
			else
				fprintf(fp, "%.3lfsec", time / 1000000000.0);
		}
		void Print()
		{
			fprintf(outFile, "%s :: ", printString);

			std::chrono::nanoseconds n = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - pt);

			long long int val = n.count();
			PrintTime(outFile, val);
			fprintf(outFile, "\n");

			printed = true;
		}
		~TimeProfiler()
		{
			if (!printed)
				Print();

			if (printString)
				free(printString);                 
		}
	};
	struct AccumulatedTimeProfiler
	{
		bool started;
		long long int accumulated, tmin, tmax, count;
		std::chrono::system_clock::time_point pt;
		const char* printString;

		AccumulatedTimeProfiler(const char* string) : printString(string), accumulated(0), count(0), started(false), tmin(LONG_MAX), tmax(0)
		{}

		void StartProfile()
		{
			started = true;
			pt = std::chrono::system_clock::now();
		}
		void EndProfileAndAccumulate()
		{
			if (started)
			{
				std::chrono::nanoseconds n = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - pt);
				if (count == 0)
				{
					tmin = tmax = n.count();
				}
				else
				{
					tmin = min(tmin, n.count());
					tmax = max(tmax, n.count());
				}
				started = false;
				count++;
				accumulated += n.count();
			}
		}

		static void PrintTime(FILE* fp, const long long int& time)
		{
			if (time < 1000)
				fprintf(fp, "%lldnsec", time);
			else if (time < 1000 * 1000)
				fprintf(fp, "%.3lfusec", time / 1000.0);
			else if (time < 1000 * 1000 * 1000)
				fprintf(fp, "%.3lfmsec", time / 1000000.0);
			else 
				fprintf(fp, "%.3lfsec", time / 1000000000.0); 
		}

		void Print(FILE* fp)
		{
			fprintf(fp, "%s, ", printString);
			fprintf(fp, "Total time :: ");
			PrintTime(fp, accumulated);
			fprintf(fp, ", Counted :: %d", count);
			fprintf(fp, ", Min time :: ");
			PrintTime(fp, tmin);
			fprintf(fp, ", Max time :: ");
			PrintTime(fp, tmax);
			fprintf(fp, ", Avg time :: ");
			PrintTime(fp, count != 0 ? accumulated / count : 0);
			fprintf(fp, "\n");
		}
		void Print()
		{
			Print(GetFilePtr());
		}
	};
#else
#define TimeProfiler(x)
#define TimeProfiler(x, y)
#define AccumulatedTimeProfiler(x)
#endif
}