// export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/usr/local/cuda/lib64

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#include <sys/time.h>

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/extrema.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
using namespace std;

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

#define NUM_THREADS 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


typedef pair<float, float> city_t;

const float CROSSOVER_RATE = 1.0;
const float MUTATION_RATE = 0.1;

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

double read_timer()
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}


__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

struct RandomNumberFunctor : 
    public thrust::unary_function<unsigned int, float>
{
    unsigned int mainSeed;
    RandomNumberFunctor(unsigned int _mainSeed) : 
        mainSeed(_mainSeed) {}

    __host__ __device__
        float operator()(unsigned int threadIdx) 
    {
        unsigned int seed = hash(threadIdx) * mainSeed;

        // seed a random number generator
        thrust::default_random_engine rng(seed);

        // create a mapping from random numbers to [0,1)
        thrust::uniform_real_distribution<float> u01(0,1);

        return u01(rng);
    }
};

// Each gene has m chromosomes
void initialize_genes(thrust::host_vector<int> &genes, int m, int n)
{
    for (int i = 0; i < n; i++) 
    {
        // traversal sequence
        thrust::sequence(genes.begin() + (i * m), genes.begin() + (i * m) + m, 0, 1);

        for (int j = 0; j < m; j++)
        {
            int r = rand() % m;

            swap(genes[(i * m) + j], genes[(i * m) + r]);
        }
    }
}

void print_genes(thrust::host_vector<int> &genes, thrust::device_vector<int> &d_genes, int m, int n)
{
    thrust::copy(d_genes.begin(), d_genes.end(), genes.begin());

    printf("\n");

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < m; j++) 
        {
            printf("%d ", genes[i * m + j]);
        }

        printf("\n");
    }

    printf("\n");
}

void compute_distances(thrust::host_vector< city_t > &cities, thrust::host_vector<float> &distances, int m)
{
    thrust::fill(distances.begin(), distances.end(), 0.0);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float x = cities[i].first - cities[j].first,
                  y = cities[i].second - cities[j].second;

            distances[i * m + j] = sqrt(x * x + y * y);
        }
    }
}

__host__ __device__ float individual_distance(float *distances, int *current_gene, int m)
{
    float total_distance = 0.0;

    for (int i = 1; i < m; i++)
    {
        total_distance += distances[current_gene[i - 1] * m + current_gene[i]];
    }

    return total_distance + distances[current_gene[m - 1] * m + current_gene[0]];
}

__global__ void compute_fitness(float *fitness, float *distances, int *genes, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) 
    {
        return;
    }

    fitness[tid] = 1.0 / individual_distance(distances, &genes[tid * m], m);
}

// Tournament selection operator
__global__ void selection(float *fitness, city_t *cities, int *genes, int *offspring, float *random, int *selections, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) 
    {
        return;
    }

    int vs = random[tid] * n;

    selections[tid] = fitness[tid] > fitness[vs] ? tid : vs;
}

// 2-point PMX crossover operator
__global__ void crossover(int *genes, int *offspring, float *random, int *selections, int *reverse_index, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x,
        random_block = tid * 3;

    tid *= 2;

    if (tid >= n) 
    {
        return;
    }

    // This kernel needs two random numbers, one for the random position of the crossover
    // and another for the crossover rate comparison

    int *parent1        = &genes[selections[tid + 0] * m],
        *parent2        = &genes[selections[tid + 1] * m],
        *offspring1     = &offspring[(tid + 0) * m],
        *offspring2     = &offspring[(tid + 1) * m],
        *reverse_index1 = &reverse_index[(tid + 0) * m],
        *reverse_index2 = &reverse_index[(tid + 1) * m];

    memcpy(offspring1, parent1, sizeof(int) * m);
    memcpy(offspring2, parent2, sizeof(int) * m);

    if (random[random_block] > CROSSOVER_RATE)
    {
        return;
    }

    int crossover_point1 = random[random_block + 1] * m,
        crossover_point2 = random[random_block + 2] * m;

    int temp;

    if (crossover_point1 > crossover_point2) 
    {
        temp = crossover_point1;
        crossover_point1 = crossover_point2;
        crossover_point2 = temp;
    }

    // Compute reverse index
    // (avoids n^2 search)
    for (int i = 0; i < m; i++) 
    {
        reverse_index1[offspring1[i]] = i;
        reverse_index2[offspring2[i]] = i;
    }

    // for (int i = 0; i < m; i++) 
    // {
    //     cuPrintf("%d\n", offspring1[i]);
    // }
    // cuPrintf("\n");
    // for (int i = 0; i < m; i++) 
    // {
    //     cuPrintf("%d\n", reverse_index1[i]);
    // }
    // cuPrintf("\n");

    for (int i = crossover_point1; i <= crossover_point2; i++)
    {
        // TODO: Optimize this N^2 search
        // int j;
        // for (j = 0; j < m; j++)
        // {
        //     if (offspring1[j] == parent2[i])
        //     {
        //         // temp = offspring1[i];
        //         // offspring1[i] = offspring1[j];
        //         // offspring1[j] = temp;
        //         break;
        //     }
        // }

        // if (j != reverse_index1[parent2[i]])
        // {
        //     cuPrintf("1: %d != %d\n", j, reverse_index1[parent2[i]]);
        //     // int *hats = (int*)0xffffffff;
        //     // *hats = 12;
        // }

        int j = reverse_index1[parent2[i]];

        temp = offspring1[i];
        offspring1[i] = offspring1[j];
        offspring1[j] = temp;

        // Swap reverse index
        temp = reverse_index1[offspring1[j]];
        reverse_index1[offspring1[j]] = reverse_index1[offspring1[i]];
        reverse_index1[offspring1[i]] = temp;
    }

    for (int i = crossover_point1; i <= crossover_point2; i++)
    {
        // TODO: Optimize this N^2 search
        // int j;
        // for (j = 0; j < m; j++)
        // {
        //     if (offspring2[j] == parent1[i])
        //     {
        // //         temp = offspring2[i];
        // //         offspring2[i] = offspring2[j];
        // //         offspring2[j] = temp;
        //         break;
        //     }
        // }

        // if (j != reverse_index2[parent1[i]])
        // {
        //     cuPrintf("2: %d != %d\n", j, reverse_index2[parent1[i]]);
        //     // int *hats = (int*)0xffffffff;
        //     // *hats = 12;
        // }

        int j = reverse_index2[parent1[i]];

        temp = offspring2[i];
        offspring2[i] = offspring2[j];
        offspring2[j] = temp;

        // Swap reverse index
        temp = reverse_index2[offspring2[j]];
        reverse_index2[offspring2[j]] = reverse_index2[offspring2[i]];
        reverse_index2[offspring2[i]] = temp;
        // temp = reverse_index2[j];
        // reverse_index2[j] = reverse_index2[i];
        // reverse_index2[i] = temp;
    }
}

__global__ void mutation(int *offspring, float *random, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) 
    {
        return;
    }

    if (random[tid] > MUTATION_RATE)
    {
        return;
    }

    int *target = &offspring[tid * m],
        mutation_point1 = random[(tid + 1) % n],
        mutation_point2 = random[(tid + 2) % n],
        temp = target[mutation_point1];

    target[mutation_point1] = target[mutation_point2];
    target[mutation_point2] = temp;
}

void read_cities(istream &in, thrust::host_vector<city_t> &cities) 
{
    float x, y;
    int m;

    in >> m;

    cities.resize(m);

    for (int i = 0; i < m; i++)
    {
        in >> x >> y;
        cities[i] = make_pair(x, y);
        cout << cities[i].first << " " << cities[i].second << endl;
    }
}

int main(int argc, char **argv)
{    
    unsigned int seed = time(NULL);

    srand(time(NULL));

    cudaDeviceSynchronize(); 

    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of genes in the population\n");
        printf("-s <int> to set the number of generations to be simulated\n");
        printf("-i <filename> to specify the input file name\n");
        printf("-o <filename> to specify the output file name\n");

        return 0;
    }
    
    int n = read_int(argc, argv, "-n", 1 << 12),
        steps = read_int(argc, argv, "-s", 1000);

    char *input_file = read_string(argc, argv, "-i", "default.in");
    char *output_file = read_string(argc, argv, "-o", "result-gpu.txt");

    ifstream input(input_file);
    ofstream output(output_file);

    thrust::host_vector<city_t> cities;

    read_cities(input, cities);

    int m = cities.size();

    // GPU genes data structure
    int *d_genes_raw, *d_offspring_raw, *d_selections_raw, *d_reverse_index_raw;
    float *d_fitness_raw, *d_random_raw, *d_distances_raw;
    city_t *d_cities_raw;

    thrust::host_vector<int> genes(m * n);
    thrust::host_vector<int> offspring(m * n);
    thrust::host_vector<float> distances(m * m);
    thrust::host_vector<float> fitness(n);
    thrust::host_vector<float> random(4 * n);
    thrust::host_vector<int> selections(n);

    thrust::device_vector<int> d_genes;
    thrust::device_vector<int> d_offspring;
    thrust::device_vector<float> d_distances;
    thrust::device_vector<int> d_reverse_index(m * n * 2);
    thrust::device_vector<city_t> d_cities;
    thrust::device_vector<float> d_fitness;
    thrust::device_vector<float> d_random(4 * n);
    thrust::device_vector<int> d_selections(n);

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

    initialize_genes(genes, m, n);
    compute_distances(cities, distances, m);

    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //     {
    //         cout << setprecision(2) << fixed << distances[i * m + j] << " ";
    //     }
    //     cout << endl;

    // }

    // cout << "TEST" << endl;
    // int test_gene[1000];

    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //     {
    //         test_gene[j] = (i + j) % m;
    //         cout << test_gene[j] << " ";
    //     }
    //     cout << ": " << individual_distance(&distances[0], test_gene, m) << endl;
    // }

    d_genes = genes;
    d_offspring = offspring;
    // d_cities = cities;
    d_fitness = fitness;
    d_distances = distances;

    // print_genes(genes, d_genes, m, n);

    d_fitness_raw = thrust::raw_pointer_cast(&d_fitness[0]);
    d_genes_raw = thrust::raw_pointer_cast(&d_genes[0]);
    d_distances_raw = thrust::raw_pointer_cast(&d_distances[0]);
    // d_cities_raw = thrust::raw_pointer_cast(&d_cities[0]);
    d_reverse_index_raw = thrust::raw_pointer_cast(&d_reverse_index[0]);

    double simulation_time = read_timer();

    cudaPrintfInit ();
    for (int step = 0; step < steps; step++)
    {
        // Compute fitness for all genes
        compute_fitness <<< blks, NUM_THREADS >>> (d_fitness_raw, d_distances_raw, d_genes_raw, m, n);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Normalize fitness (sum of fitness = 1)
        float total_fitness = thrust::reduce(d_fitness.begin(), d_fitness.end());
        thrust::constant_iterator<float> normalization(total_fitness);
        thrust::transform(d_fitness.begin(), d_fitness.end(), normalization, d_fitness.begin(), thrust::divides<float>());

        // Selection
        d_offspring_raw = thrust::raw_pointer_cast(&d_offspring[0]);
        d_random_raw = thrust::raw_pointer_cast(&d_random[0]);
        d_selections_raw = thrust::raw_pointer_cast(&d_selections[0]);

        // Generate 4n random numbers
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(4 * n),
                          d_random.begin(), 
                          RandomNumberFunctor(seed));

        // Uses n random numbers
        selection <<< blks, NUM_THREADS >>> (d_fitness_raw, d_cities_raw, d_genes_raw, d_offspring_raw, d_random_raw, d_selections_raw, m, n);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        d_random_raw = thrust::raw_pointer_cast(&d_random[n]);  

        // Uses 2n random numbers
        crossover <<< blks, NUM_THREADS >>> (d_genes_raw, d_offspring_raw, d_random_raw, d_selections_raw, d_reverse_index_raw, m, n);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        d_random_raw = thrust::raw_pointer_cast(&d_random[3 * n]);
        
        // Mutation
        // Uses n random numbers
        mutation <<< blks, NUM_THREADS >>> (d_offspring_raw, d_random_raw, m, n);

        swap(d_genes, d_offspring);

        // print_genes(offspring, d_offspring, m, n);
        
        // thrust::device_vector<int> temp = d_genes;
        // d_genes = d_offspring;
        // d_offspring = d_genes;

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    cudaPrintfDisplay (stdout, true);
    cudaPrintfEnd ();

    simulation_time = read_timer() - simulation_time;
    cout << simulation_time;

    // print_genes(offspring, d_offspring, m, n);

    compute_fitness <<< blks, NUM_THREADS >>> (d_fitness_raw, d_distances_raw, d_genes_raw, m, n);
    thrust::copy(d_fitness.begin(), d_fitness.end(), fitness.begin());
    thrust::copy(d_genes.begin(), d_genes.end(), genes.begin());
    thrust::copy(d_offspring.begin(), d_offspring.end(), offspring.begin());

    int the_best = thrust::distance(fitness.begin(), thrust::max_element(fitness.begin(), fitness.begin() + n));
    cout << "OMG: " << the_best << " : " << (1.0 / fitness[the_best]) << endl;
    output << (1.0 / fitness[the_best]) << endl;
    for (int i = 0; i < m; i++)
    {
        output << offspring[the_best * m + i] << " ";
    }

    return 0;
}
