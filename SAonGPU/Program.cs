using SharpDX;
using SharpDX.D3DCompiler;
using SharpDX.Direct3D;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using System.Diagnostics;

var device = new SharpDX.Direct3D11.Device(DriverType.Hardware, DeviceCreationFlags.None);
var context = device.ImmediateContext;

var computeShaderByteCode = ShaderBytecode.CompileFromFile("annealKernel.hlsl", "CSMain", "cs_5_0");
var computeShader = new ComputeShader(device, computeShaderByteCode);
context.ComputeShader.Set(computeShader);

int width = 16;
int height = 16;
int[] initialLattice = new int[width * height];
Random rand = new Random();

for (int i = 0; i < width * height; i++)
{
    initialLattice[i] = -1;// rand.Next(2) * 2 - 1; // will generate either -1 or 1
}

var stagingTexture = new Texture2D(device, new Texture2DDescription
{
    Width = width,
    Height = height,
    MipLevels = 1,
    ArraySize = 1,
    Format = Format.R32_SInt,
    Usage = ResourceUsage.Staging,
    BindFlags = BindFlags.None,
    CpuAccessFlags = CpuAccessFlags.Write,
    SampleDescription = new SampleDescription(1, 0)
});

// Lock the staging buffer, copy the data and then unlock it
DataBox mappedResource = context.MapSubresource(stagingTexture, 0, MapMode.Write, SharpDX.Direct3D11.MapFlags.None);
unsafe
{
    int* ptr = (int*)mappedResource.DataPointer;
    for (int i = 0; i < initialLattice.Length; i++)
    {
        *(ptr + i) = initialLattice[i];
    }
}
context.UnmapSubresource(stagingTexture, 0);

// Create the main lattice texture
var lattice = new Texture2D(device, new Texture2DDescription
{
    Width = width,
    Height = height,
    MipLevels = 1,
    ArraySize = 1,
    Format = Format.R32_SInt,
    Usage = ResourceUsage.Default,
    BindFlags = BindFlags.UnorderedAccess,
    CpuAccessFlags = CpuAccessFlags.None,
    SampleDescription = new SampleDescription(1, 0)
});

// Copy from the staging texture to the main texture
context.CopyResource(stagingTexture, lattice);
stagingTexture.Dispose();

var uav = new UnorderedAccessView(device, lattice);
context.ComputeShader.SetUnorderedAccessView(0, uav);

var magnetizationBuffer = new SharpDX.Direct3D11.Buffer(device, new BufferDescription
{
    BindFlags = BindFlags.UnorderedAccess,
    Usage = ResourceUsage.Default,
    CpuAccessFlags = CpuAccessFlags.None,
    OptionFlags = ResourceOptionFlags.BufferStructured,
    StructureByteStride = sizeof(int),
    SizeInBytes = sizeof(int)
});

var magnetizationUAV = new UnorderedAccessView(device, magnetizationBuffer);
context.ComputeShader.SetUnorderedAccessView(1, magnetizationUAV);

var sw = new Stopwatch();
sw.Restart();

context.Dispatch(16, 16, 1);


var magnetizationStaging = new SharpDX.Direct3D11.Buffer(device, new BufferDescription
{
    Usage = ResourceUsage.Staging,
    BindFlags = BindFlags.None,
    CpuAccessFlags = CpuAccessFlags.Read,
    OptionFlags = ResourceOptionFlags.BufferStructured,
    StructureByteStride = sizeof(int),
    SizeInBytes = sizeof(int)
});

context.CopyResource(magnetizationBuffer, magnetizationStaging);

var mappedResource2 = context.MapSubresource(magnetizationStaging, 0, MapMode.Read, SharpDX.Direct3D11.MapFlags.None);


sw.Stop();
Console.WriteLine(sw.ElapsedMilliseconds);

int magnetization;
unsafe
{
    magnetization = *((int*)mappedResource2.DataPointer.ToPointer());
}
context.UnmapSubresource(magnetizationStaging, 0);

Console.WriteLine($"Magnetization: {magnetization}");

sw.Restart();
for (int i = 0; i < initialLattice.Length; i++)
{
    for (int j = 0;j < 100000; j++)
    {
        initialLattice[i]++;
    }
}

sw.Stop();
Console.WriteLine(sw.ElapsedMilliseconds);
Console.WriteLine(initialLattice.Sum());
for (int i = 0; i < width * height; i++)
{
    initialLattice[i] = -1;// rand.Next(2) * 2 - 1; // will generate either -1 or 1
}
sw.Restart();

Parallel.For( 0, initialLattice.Length, i =>
{
    {
        for (int j = 0; j < 100000; j++)
        {
            initialLattice[i]++;
        }
    }
});

sw.Stop();
Console.WriteLine(sw.ElapsedMilliseconds);
Console.WriteLine(initialLattice.Sum());


// Cleanup
uav.Dispose();
lattice.Dispose();
magnetizationUAV.Dispose();
magnetizationBuffer.Dispose();
magnetizationStaging.Dispose();
computeShader.Dispose();
computeShaderByteCode.Dispose();
context.Dispose();
device.Dispose();