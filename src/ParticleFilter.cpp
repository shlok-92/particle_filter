#include "ros/ros.h"
#include "std_msgs/String.h"
#include <vector>
#include <stdlib.h>
#include <stdexcept>
#include <math.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
float landmark[4][2]={{20.0,20.0},{80.0,80.0},{20.0,80.0},{80.0,20.0}};
int num_particles=1000;
int num_landmark=4;
int world_size=100;
Mat image_= Mat::zeros(100, 100, CV_32FC3);


class ParticleFilter
{
public:
    // Data Members

    float x;
    float y;
    float orientation;
    float forward_noise;
    float turn_noise;
    float sense_noise;
    unsigned seed;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution_sense_noise;
    std::normal_distribution<double> distribution_forward_noise;
    std::normal_distribution<double> distribution_turn_noise;



    //Member Functions
    ParticleFilter();
    void set(float x, float y, float orientation);
    void set_noise(float forward_noise, float turn_noise, float sense_noise);
    std::vector<float> sense();
    void move(ParticleFilter pf,float turn, float forward);
    float Gaussian(float x, float mean, float variance);
    float measurement_prob(std::vector<float> measurement);

};
ParticleFilter::ParticleFilter()
{
    x=rand()%(world_size) + 1.0;
    y=rand()%(world_size) + 1.0;
    orientation=2.0*3.14*float((rand()%11))/100.0;
    forward_noise=0.0;
    turn_noise=0.0;
    sense_noise=0.0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    distribution_sense_noise = std::normal_distribution<double>(0.0,sense_noise);
    distribution_forward_noise = std::normal_distribution<double>(0.0,forward_noise);
    distribution_turn_noise = std::normal_distribution<double>(0.0,turn_noise);


}
void ParticleFilter::set(float _x, float _y, float _orientation)
{
    //Checking bounds before assigning values

    if(_x<0 || _x>world_size)
    {
        std::cout<<"X value out of bound"<<"\n";
        throw std::invalid_argument( "X value out of bound" );
    }
    else
    {
        x=_x;
    }

    if(_y<0 || _y>world_size)
    {
        std::cout<<"Y value out of bound"<<"\n";
        throw std::invalid_argument( "Y value out of bound" );
    }
    else
    {
        y=_y;
    }
    if(_orientation<0 || _orientation>2*3.14)
    {
        std::cout<<"orientation value out of bound"<<"\n";
        throw std::invalid_argument( "orientation value out of bound" );
    }
    else
    {
        orientation=_orientation;
    }

}
void ParticleFilter::set_noise(float _forward_noise, float _turn_noise, float _sense_noise)
{
    forward_noise=float(_forward_noise);
    turn_noise=float(_turn_noise);
    sense_noise=float(_sense_noise);

    distribution_sense_noise = std::normal_distribution<double>(0.0,sense_noise);
    distribution_forward_noise = std::normal_distribution<double>(0.0,forward_noise);
    distribution_turn_noise = std::normal_distribution<double>(0.0,turn_noise);
}
std::vector<float> ParticleFilter::sense()
{
    std::vector<float> dist2landmark;
    float dist;
    for (int i = 0; i < num_landmark; ++i) {
        dist=sqrt(pow(x-landmark[i][0],2)+pow(y-landmark[i][1],2));
        dist+=distribution_sense_noise(generator);
        dist2landmark.push_back(dist);
    }
    return dist2landmark;
}
float ParticleFilter::Gaussian(float mean, float variance,float x)
{
    return exp(- (pow((mean - x),2)) / (pow(variance,2)*2.0)) / sqrt(2.0 * M_PI * pow(variance,2));
}
void ParticleFilter::move(ParticleFilter pf,float turn, float forward)
{
    if (forward < 0)
        throw std::invalid_argument( "Robot cannot go backward" );

    orientation = orientation + float(turn)+distribution_turn_noise(generator);
    orientation = fmod(orientation,2.0*3.14);

    float dist = float(forward)+distribution_forward_noise(generator);;
    x = x + (cos(orientation) * dist);
    y = y + (sin(orientation) * dist);

    if(x>0)
        x = fmod(x,world_size);
    else
        x= world_size+x;

    if(y>0)
        y = fmod(y,world_size);
    else
        y= world_size+y;

    pf.set(x, y, orientation);
    pf.set_noise(forward_noise, turn_noise, sense_noise);
}
void print(ParticleFilter r,ParticleFilter pf)
{
    std::cout<<"Actual: "<<r.x<<"\t"<<r.y<<"\t"<<"Particle: "<<pf.x<<"\t"<<pf.y<<"\n";

}
float ParticleFilter::measurement_prob(std::vector<float> measurement)
{
    float prob = 1.0;
    float dist;
    for (int i = 0; i < num_landmark; ++i) {
        dist=sqrt(pow(x-landmark[i][0],2)+pow(y-landmark[i][1],2));
        prob *= Gaussian(dist, sense_noise, measurement[i]);
    }
    return prob;
}
float eval(ParticleFilter r,ParticleFilter p[])
{
    float sum = 0.0;
    float dx,dy,err;
    for (int i = 0; i < num_particles; ++i) {
        if((p[i].x - r.x + (world_size/2.0))>0)
        {
            dx = fmod((p[i].x - r.x + (world_size/2.0)) , world_size) - (world_size/2.0);
        }
        else
        {
            dx=world_size+(p[i].x - r.x + (world_size/2.0))- (world_size/2.0);
        }

        if((p[i].y - r.y + (world_size/2.0))>0)
        {
            dy = fmod((p[i].y - r.y + (world_size/2.0)) , world_size) - (world_size/2.0);
        }
        else
        {
            dy=world_size+(p[i].y - r.y + (world_size/2.0))- (world_size/2.0);
        }

        err = sqrt(dx * dx + dy * dy);
        sum += err  ;
    }

    return sum / float(num_particles);
}

void visualization(ParticleFilter r, ParticleFilter pf[])
{
    Mat image_= Mat::zeros(100, 100, CV_32FC3);
    namedWindow( "Map", CV_WINDOW_NORMAL);
    resizeWindow("Map", 1600, 1600);

    image_.at<Point3f>(r.x, r.y) = Point3f(0, 1, 0);
    for (int i = 0; i < num_particles; ++i) {
        image_.at<Point3f>(pf[i].x, pf[i].y) = Point3f(1, 0, 0);
    }
    if (! (image_.empty()) )
    {

        imshow("Map", image_);
    }
    waitKey(10000);
}

int main()
{

    std::uniform_real_distribution<double> random2;
    random2=std::uniform_real_distribution<double>(0.0,1.0);
    std::random_device rand_dev;
    std::mt19937_64 init_generator(rand_dev());

    ParticleFilter pf[num_particles];
    ParticleFilter pf2[num_particles];

    ParticleFilter r;
    r.set_noise(0.05,0.05,5.0);

    for (int z = 0; z < num_particles; ++z) {
        pf[z].set_noise(0.05,0.05,5.0);
    }
    std::vector<float> w;
    //loop begins
    for (int var = 0; var <100 ; ++var) {

        r.move(r,0.1,5.0);

        for (int z = 0; z < num_particles; ++z) {
            pf[z].move(pf[z],0.1,5.0);
        }

        for (int i = 0; i < num_particles; ++i) {
            w.push_back(pf[i].measurement_prob(r.sense()));
        }

        int index=random2(init_generator)*num_particles;;
        float max=*std::max_element(std::begin(w),std::end(w));

        float beta=0.0;


        for (int i = 0; i < num_particles; ++i) {
            beta+=random2(init_generator)*2.0*max;
            while(beta>w[index])
            {
                beta-=w[index];
                index=(index+1)%num_particles;
            }
            pf2[i]=pf[index];
        }
        for (int j = 0; j < num_particles; ++j) {
            pf[j]=pf2[j];
            print(r,pf[j]);
        }

       // std::cout<<eval(r,pf)<<"\n";
        visualization(r,pf);
    }
    return 0;
}
