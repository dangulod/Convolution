#include <iostream>

using namespace std;

extern float * poipar(float lambda, float xi, float psi, float u, int n);

int main(int argc, char** argv)
{
    int n = 1e6;

    float * out = poipar(100, 1, 2, 1, n);

    for(int i = 0; i < n; i++)
    {
        printf("%f\n", out[i]);
    }

    return 0;
}
