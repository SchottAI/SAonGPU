RWTexture2D<int> lattice : u0;
RWStructuredBuffer<int> magnetization : u1;  // Store magnetization

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    int x = id.x;
    int y = id.y;

    for (int i = 0; i < 100000; i++)
    {
       

        //int spin = lattice[int2(x, y)];

        //int sum = lattice[int2((x + 1) % 256, y)] +
        //    lattice[int2((x - 1 + 256) % 256, y)] +
        //    lattice[int2(x, (y + 1) % 256)] +
        //    lattice[int2(x, (y - 1 + 256) % 256)];

        //int dE = 2 * spin * sum;

        //if (dE < 0)
        {
            //lattice[int2(x, y)] = -spin;
            lattice[int2(x, y)] += 1;
        }
    }
    //InterlockedAdd(magnetization[0], lattice[int2(x, y)]);
}