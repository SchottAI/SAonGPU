using ManagedCuda;
using ManagedCuda.BasicTypes;
//using ManagedCuda.CudaCompiler;
using ManagedCuda.VectorTypes;
using System;

        int N = 1024; // Number of parallel annealing simulations
        float initialTemp = 100.0f;
        float coolingRate = 0.995f;
        int maxIterations = 1000;

        CudaContext ctx = new CudaContext();
        CudaKernel annealKernel = ctx.LoadKernel("annealKernel.ptx", "annealKernel");

        float[] hostCurrentSolutions = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostCurrentSolutions[i] = 10.0f * ((float)new Random().NextDouble() - 0.5f); // Random start between -5 and 5
        }

        CudaDeviceVariable<float> devCurrentSolutions = hostCurrentSolutions;
        CudaDeviceVariable<float> devNextSolutions = new CudaDeviceVariable<float>(N);

        annealKernel.GridDimensions = (N + 255) / 256;
        annealKernel.BlockDimensions = 256;

        float temperature = initialTemp;
        for (int iter = 0; iter < maxIterations; iter++)
        {
            annealKernel.Run(devCurrentSolutions.DevicePointer, devNextSolutions.DevicePointer, temperature, N);

            // Swap current and next
            var temp = devCurrentSolutions;
            devCurrentSolutions = devNextSolutions;
            devNextSolutions = temp;

            temperature *= coolingRate;
        }

        devCurrentSolutions.CopyToHost(hostCurrentSolutions);
        for (int i = 0; i < N; i++)
        {
            Console.WriteLine($"Solution {i}: {hostCurrentSolutions[i]}");
        }
