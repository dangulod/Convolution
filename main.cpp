#include <iostream>
#include <chrono>


using namespace std;

extern float * poipar(float lambda, float xi, float psi, float u, int n);

int main(int argc, char** argv)
{
    int n = 1e6;

    auto dx = std::chrono::high_resolution_clock::now();

    float * out = poipar(1e3, 1, 2, 1, n);

    auto dy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dif = dy - dx;
    std::cout << dif.count() << " seconds" << std::endl;


    for(int i = 0; i < 10; i++)
    {
        printf("%f\n", out[i]);
    }

    return 0;
}
